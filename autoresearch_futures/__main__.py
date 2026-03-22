# autoresearch_futures/__main__.py
"""
Main entry point for autoresearch-futures.
Run with: python -m autoresearch_futures [command]
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch-futures: Self-evolving futures strategy research"
    )
    parser.add_argument(
        "command",
        choices=["prepare", "evolve", "test", "monitor"],
        help="Command to run",
    )
    parser.add_argument("--symbols", nargs="+", help="Symbols to process (default: all)")

    args = parser.parse_args()

    if args.command == "prepare":
        from autoresearch_futures.prepare import (
            download_all_contracts,
            save_splits,
            generate_walk_forward_splits,
            list_available_symbols,
            load_data,
        )

        print("Downloading futures data...")
        download_all_contracts(symbols=args.symbols)

        symbols = list_available_symbols()
        if symbols:
            df = load_data(symbols[0])
            if df is not None and len(df) > 0:
                start = df["datetime"].min().strftime("%Y-%m-%d")
                end = df["datetime"].max().strftime("%Y-%m-%d")
                splits = generate_walk_forward_splits(start, end)
                save_splits(splits)
                print(f"Generated {len(splits)} walk-forward splits")
        print("Done!")

    elif args.command == "evolve":
        from autoresearch_futures.evolve import run_evolution_step, log_results
        from autoresearch_futures.prepare import load_splits, load_split_data, list_available_symbols
        from autoresearch_futures.config import DEFAULT_PARAMS
        import subprocess

        splits = load_splits()
        if not splits:
            print("No splits found. Run 'python -m autoresearch_futures prepare' first.")
            sys.exit(1)

        symbols = args.symbols or list_available_symbols()
        if not symbols:
            print("No symbols available. Run 'python -m autoresearch_futures prepare' first.")
            sys.exit(1)

        tick_sizes = {s: 1.0 for s in symbols}
        contract_multipliers = {s: 10 for s in symbols}

        try:
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        except subprocess.CalledProcessError:
            commit = "unknown"

        split = splits[0] if splits else {}
        if not split:
            print("No splits available.")
            sys.exit(0)

        print(f"Running evolution on split {split['split_id']}...")

        # Load data for each symbol, filter out those with no data
        data_dict = {}
        for symbol in symbols:
            df = load_split_data(symbol, split, "train")
            if df is not None and not df.empty:
                data_dict[symbol] = df
            else:
                print(f"Warning: No data for {symbol}, skipping")

        if not data_dict:
            print("No data available for any symbol. Run 'prepare' first.")
            sys.exit(1)

        score, results, signals = run_evolution_step(
            data_dict, DEFAULT_PARAMS, tick_sizes, contract_multipliers,
        )

        print("---")
        print(f"score:    {score:.6f}")
        for symbol, result in results.items():
            print(f"{symbol}: sharpe={result.sharpe_ratio:.4f} return={result.net_return:.4f}")

        log_results(split_id=split["split_id"], commit=commit, score=score,
                   results=results, description="experiment run")

    elif args.command == "test":
        import subprocess
        result = subprocess.run(["pytest", "tests/", "-v"])
        sys.exit(result.returncode)

    elif args.command == "monitor":
        from autoresearch_futures.monitor import run_monitor
        from autoresearch_futures.config import DEFAULT_PARAMS

        symbols = args.symbols or ["rb", "i"]
        print(f"启动实时监控: {symbols}")
        print("按 Ctrl+C 停止")
        print("-" * 50)

        try:
            run_monitor(
                symbols=symbols,
                params=DEFAULT_PARAMS,
                poll_interval=15,
                continuous=True,
            )
        except KeyboardInterrupt:
            print("\n监控已停止")


if __name__ == "__main__":
    main()