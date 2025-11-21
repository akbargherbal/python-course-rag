"""
Command-line interface for course semantic search.
"""
from pathlib import Path
from src.query_engine import CourseQueryEngine
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

class CourseSearchCLI:
    """CLI for interactive course search."""

    def __init__(self, db_path: Path = Path("./data/lancedb")):
        """Initialize CLI with query engine."""
        with console.status("[bold green]Loading search engine...[/bold green]"):
            self.engine = CourseQueryEngine(db_path=db_path, use_reranking=True)

    def format_result(self, result: dict, index: int) -> Panel:
        """Format single search result for display."""
        content = f"""
[bold]📍 Section:[/bold] {result['section']}
[bold]⏱️  Timestamp:[/bold] {result['start_timestamp']} - {result['end_timestamp']}
[bold]📊 Score:[/bold] {result['score']:.3f}

[italic]{result['text_snippet']}...[/italic]
"""
        return Panel(
            content,
            title=f"[bold cyan]{index}. {result['video_title']}[/bold cyan]",
            border_style="blue"
        )

    def search(self, query: str, limit: int = 5):
        """Execute search and display results."""
        console.print(f"\n[bold]🔍 Searching for:[/bold] [yellow]{query}[/yellow]\n")

        try:
            result = self.engine.query(query, top_k=limit, verbose=False)

            if not result['results']:
                console.print("[red]❌ No results found[/red]")
                return

            # Display results
            for i, res in enumerate(result['results'], 1):
                console.print(self.format_result(res, i))

            # Print copy-paste friendly summary
            console.print("\n[bold]📋 Quick Reference:[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("ID", style="dim")
            table.add_column("Title")
            table.add_column("Time", style="green")
            
            for res in result['results']:
                table.add_row(
                    str(res['video_id']),
                    res['video_title'],
                    res['start_timestamp']
                )
            console.print(table)
            console.print(f"\n[dim]Found {result['total_results']} results in {result['query_time']:.2f}s[/dim]\n")

        except Exception as e:
            console.print(f"[bold red]❌ Error:[/bold red] {e}")

@click.group()
def cli():
    """Python Course Semantic Search CLI."""
    pass

@cli.command()
@click.option('--query', '-q', prompt='Enter search query', help='Search query')
@click.option('--limit', '-l', default=5, help='Number of results to show')
@click.option('--db-path', default='./data/lancedb', help='Path to LanceDB')
def search(query: str, limit: int, db_path: str):
    """Search course content."""
    cli_engine = CourseSearchCLI(db_path=Path(db_path))
    cli_engine.search(query, limit=limit)

@cli.command()
@click.option('--db-path', default='./data/lancedb', help='Path to LanceDB')
def interactive(db_path: str):
    """Start interactive search mode."""
    cli_engine = CourseSearchCLI(db_path=Path(db_path))

    console.print(Panel.fit(
        "[bold yellow]🎓 Python Course Search - Interactive Mode[/bold yellow]\n"
        "[dim]Type 'exit', 'quit', or 'q' to stop[/dim]"
    ))

    while True:
        try:
            query = click.prompt("\n❓ Query")

            if query.lower() in ['exit', 'quit', 'q']:
                console.print("[green]Goodbye! 👋[/green]")
                break

            if not query.strip():
                continue

            cli_engine.search(query, limit=3)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
            break

if __name__ == "__main__":
    cli()