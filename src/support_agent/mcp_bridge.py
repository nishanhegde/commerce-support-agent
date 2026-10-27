"""Day 46: mcp-commerce's tools as ordinary sync LangChain tools.

The problem: an MCP client is async (it reads and writes a subprocess's
stdin/stdout on an event loop), and our graph is sync: llm.invoke,
tool.invoke, SqliteSaver, input(). langchain-mcp-adapters solves it by making
the tools async-only, which would push async through the whole graph. This
bridge solves it at the edge instead:

    graph (main thread, sync)                 bridge thread (asyncio loop)
    -------------------------                 ----------------------------
    tool.invoke(args)
      -> run_coroutine_threadsafe(  -------->  session.call_tool(name, args)
           ..., loop).result()                   -> stdio -> mcp-commerce
      <- blocks until it's done     <--------    <- result

Java analogy: an ExecutorService with one thread, and Future.get().
asyncio has no blocking get() of its own, because blocking inside the loop
would stall the loop. Blocking from ANOTHER thread is fine, which is why the
loop gets a thread of its own.

The one non-obvious rule: stdio_client and ClientSession are async context
managers that must be entered and exited by the same task (they hold anyio
cancel scopes, which raise if a different task closes them). So the session
lives inside one long-running coroutine, `_serve`, which opens it, signals
"ready", and waits for "stop". Tool calls are separate, short tasks on the
same loop, and the session handles them concurrently.
"""

import asyncio
import os
import threading
from pathlib import Path

from langchain_core.tools import StructuredTool, ToolException
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.types import Tool

# The Month 2 repo, a sibling of this one. MCP_COMMERCE_DIR overrides it.
DEFAULT_SERVER_DIR = Path(__file__).resolve().parents[3] / "mcp-commerce"

STARTUP_TIMEOUT = 30  # seconds; the first `uv run` may need to sync a venv
CALL_TIMEOUT = 30


def mcp_commerce_server() -> StdioServerParameters:
    """How to start mcp-commerce: `uv run` inside its own repo, so it runs on
    its own venv and dependencies, exactly as Claude Desktop starts it."""
    server_dir = Path(os.environ.get("MCP_COMMERCE_DIR", DEFAULT_SERVER_DIR))
    return StdioServerParameters(
        command="uv",
        args=["run", "--directory", str(server_dir), "mcp-commerce"],
        # The SDK's default environment is a short allowlist (PATH, HOME, ...),
        # so our ANTHROPIC_API_KEY doesn't leak into a process that never
        # calls a model. MCP_TRANSPORT is forced: a .env over there set to
        # "http" would make it bind a port and never answer on stdio.
        env={**get_default_environment(), "MCP_TRANSPORT": "stdio"},
    )


class MCPBridge:
    """One MCP session, on a background event loop, usable from sync code.

        with MCPBridge(mcp_commerce_server()) as bridge:
            graph = build_explicit(tools=bridge.tools, policy=by_name(bridge.needs_approval))
    """

    def __init__(self, server: StdioServerParameters, errlog_path: Path | None = None):
        self._server = server
        self._errlog_path = errlog_path
        self._loop = asyncio.new_event_loop()
        # daemon=True: if something goes wrong and __exit__ never runs, the
        # thread won't keep the process alive at exit.
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._ready = threading.Event()
        self._session: ClientSession | None = None
        self._stop: asyncio.Event | None = None
        self._mcp_tools: list[Tool] = []
        self.tools: list[StructuredTool] = []
        self.needs_approval: set[str] = set()

    # --- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "MCPBridge":
        self._thread.start()
        self._serving = asyncio.run_coroutine_threadsafe(self._serve(), self._loop)
        self._ready.wait(STARTUP_TIMEOUT)
        if self._serving.done():
            # _serve ended before it was ready: the server failed to start.
            # .result() re-raises its exception here, in the caller's thread.
            self._serving.result()
        if self._session is None:
            raise TimeoutError(f"MCP server not ready after {STARTUP_TIMEOUT}s")
        self.tools = [self._as_langchain_tool(t) for t in self._mcp_tools]
        # The approval policy comes from the server's own annotations: a
        # write tool (readOnlyHint false or absent) waits for a human. The
        # server author decided what's safe; the client doesn't keep a list.
        self.needs_approval = {
            t.name
            for t in self._mcp_tools
            if not (t.annotations and t.annotations.read_only_hint)
        }
        return self

    def __exit__(self, *exc) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
            self._serving.result(CALL_TIMEOUT)  # waits for the session to close
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    async def _serve(self) -> None:
        errlog = open(self._errlog_path or os.devnull, "a")
        try:
            async with (
                stdio_client(self._server, errlog=errlog) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                self._mcp_tools = (await session.list_tools()).tools
                self._stop = asyncio.Event()
                self._session = session
                self._ready.set()
                await self._stop.wait()
        finally:
            self._ready.set()  # unblock __enter__ if we failed before ready
            errlog.close()

    # --- calls -------------------------------------------------------------

    def call(self, name: str, args: dict) -> str:
        """Call one MCP tool and block until it answers. Returns its text.

        An MCP error result (is_error) becomes ToolException, which the graph's
        tools node already turns into a ToolMessage with status="error".
        """
        coro = self._session.call_tool(name, args)
        result = asyncio.run_coroutine_threadsafe(coro, self._loop).result(CALL_TIMEOUT)
        text = "".join(c.text for c in result.content if c.type == "text")
        if result.is_error:
            raise ToolException(text)
        return text

    def _as_langchain_tool(self, tool: Tool) -> StructuredTool:
        # args_schema takes the server's JSON Schema as-is, so bind_tools
        # sends Claude exactly what the server declared: the same limits and
        # descriptions Claude Desktop sees. No Pydantic model in between.
        #
        # The closure is safe because `tool` is this method's parameter, a
        # fresh name per call. Written inline in a `for t in tools:` loop,
        # every closure would share `t` and call the LAST tool: Python
        # closures capture variables, not values. (Java's "effectively final"
        # rule exists to rule that out.)
        def call(**kwargs) -> str:
            return self.call(tool.name, kwargs)

        return StructuredTool(
            name=tool.name,
            description=tool.description or "",
            args_schema=tool.input_schema,
            func=call,
        )
