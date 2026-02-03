import argparse
import sys
from ingest import ingest
from rag import query

def main():
    parser = argparse.ArgumentParser(description="Document QA System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    subparsers.add_parser("ingest", help="Ingest documents from 'data' folder")

    # Query command
    query_parser = subparsers.add_parser("ask", help="Ask a question")
    query_parser.add_argument("question", type=str, help="The question to ask")

    # Interactive mode
    subparsers.add_parser("interactive", help="Start interactive chat mode")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.command == "ingest":
        ingest()
    elif args.command == "ask":
        query(args.question)
    elif args.command == "interactive":
        print("Starting interactive mode. Type 'exit' to quit.")
        while True:
            q = input("\nAsk a question: ")
            if q.lower() in ["exit", "quit", "q"]:
                break
            query(q)

if __name__ == "__main__":
    main()
