"""CLI entry point for memori.

Usage:
    memori serve --port 8080
    memori store --agent my-agent --content "Some memory"
    memori recall --agent my-agent --query "search term"
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="memori",
        description="Agent Memory Middleware CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the memori server")
    serve_parser.add_argument("--port", type=int, default=8080, help="Server port")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")

    # store
    store_parser = subparsers.add_parser("store", help="Store a memory")
    store_parser.add_argument("--agent", type=str, default="default")
    store_parser.add_argument("--content", type=str, required=True)
    store_parser.add_argument("--type", type=str, default="episodic")
    store_parser.add_argument("--url", type=str, default="http://localhost:8080")

    # recall
    recall_parser = subparsers.add_parser("recall", help="Recall memories")
    recall_parser.add_argument("--agent", type=str, default="default")
    recall_parser.add_argument("--query", type=str, required=True)
    recall_parser.add_argument("--url", type=str, default="http://localhost:8080")
    recall_parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        uvicorn.run("mnemo.api:app", host=args.host, port=args.port, reload=False)
    elif args.command == "store":
        from mnemo.sdk.python.client import MemoriClient
        client = MemoriClient(base_url=args.url, agent_id=args.agent)
        mem_id = client.store(args.content, memory_type=args.type)
        print(mem_id)
    elif args.command == "recall":
        from mnemo.sdk.python.client import MemoriClient
        client = MemoriClient(base_url=args.url, agent_id=args.agent)
        results = client.recall(args.query, top_k=args.top_k)
        for i, r in enumerate(results, 1):
            mem = r["memory"]
            print(f"{i}. [{r['score']:.3f}] {mem['content'][:120]}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
