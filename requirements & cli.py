# requirements.txt
"""
# Core dependencies
aiohttp>=3.8.0
requests>=2.28.0
numpy>=1.21.0
pandas>=1.4.0
scipy>=1.8.0
pyyaml>=6.0

# Polymarket client
py-clob-client>=0.2.0

# Kalshi client (if available)
kalshi-python>=0.1.0

# Additional utilities
python-dotenv>=0.19.0
click>=8.0.0
colorama>=0.4.4
tabulate>=0.9.0
websockets>=10.0

# Optional: For advanced features
scikit-learn>=1.1.0
plotly>=5.0.0
"""

# cli.py - Command Line Interface
import click
import asyncio
import logging
import sys
from pathlib import Path
from tabulate import tabulate
from colorama import init, Fore, Style
import yaml
from datetime import datetime

# Import our bot modules
from prediction_market_bot import MarketMakingBot, KalshiClient, PolymarketClient
from advanced_pricing_model import load_config, create_example_config, AdvancedMarketMaker

# Initialize colorama for cross-platform colored output
init()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Prediction Market Trading Bot CLI"""
    pass

@cli.command()
def init_config():
    """Initialize configuration file"""
    try:
        create_example_config()
        click.echo(f"{Fore.GREEN}✓ Created bot_config.yaml{Style.RESET_ALL}")
        click.echo(f"{Fore.YELLOW}Please edit the configuration file with your API credentials{Style.RESET_ALL}")
    except Exception as e:
        click.echo(f"{Fore.RED}✗ Error creating config: {e}{Style.RESET_ALL}")

@cli.command()
@click.option('--config', default='bot_config.yaml', help='Configuration file path')
def validate_config(config):
    """Validate configuration file"""
    try:
        config_obj = load_config(config)
        click.echo(f"{Fore.GREEN}✓ Configuration file is valid{Style.RESET_ALL}")
        
        # Display key settings
        settings_table = [
            ["Setting", "Value"],
            ["Min Profit Margin", f"{config_obj.min_profit_margin:.1%}"],
            ["Max Position Size", f"${config_obj.max_position_size}"],
            ["Update Interval", f"{config_obj.update_interval}s"],
            ["Demo Mode (Kalshi)", config_obj.kalshi_demo],
            ["Demo Mode (Polymarket)", config_obj.polymarket_demo],
        ]
        
        click.echo("\nKey Settings:")
        click.echo(tabulate(settings_table, headers="firstrow", tablefmt="grid"))
        
    except Exception as e:
        click.echo(f"{Fore.RED}✗ Configuration validation failed: {e}{Style.RESET_ALL}")

@cli.command()
@click.option('--config', default='bot_config.yaml', help='Configuration file path')
@click.option('--platform', type=click.Choice(['kalshi', 'polymarket', 'both']), default='both')
@click.option('--limit', default=10, help='Number of markets to display')
def list_markets(config, platform, limit):
    """List available markets"""
    async def _list_markets():
        try:
            config_obj = load_config(config)
            
            markets_data = []
            
            if platform in ['kalshi', 'both']:
                kalshi_client = KalshiClient(
                    email=config_obj.kalshi_email,
                    password=config_obj.kalshi_password,
                    demo=config_obj.kalshi_demo
                )
                await kalshi_client.authenticate()
                kalshi_markets = await kalshi_client.get_markets(limit=limit)
                
                for market in kalshi_markets:
                    markets_data.append([
                        "Kalshi",
                        market.get('ticker', 'N/A'),
                        market.get('title', 'N/A')[:50] + "..." if len(market.get('title', '')) > 50 else market.get('title', 'N/A'),
                        f"{market.get('yes_price', 0)/100:.2f}" if market.get('yes_price') else "N/A",
                        f"{market.get('volume', 0):,}"
                    ])
            
            if platform in ['polymarket', 'both']:
                polymarket_client = PolymarketClient(
                    private_key=config_obj.polymarket_private_key,
                    funder_address=config_obj.polymarket_funder,
                    demo=config_obj.polymarket_demo
                )
                await polymarket_client.initialize()
                poly_markets = await polymarket_client.get_markets()
                
                for market in poly_markets[:limit]:
                    tokens = market.get('tokens', [])
                    if tokens:
                        token = tokens[0]  # Take first token for display
                        markets_data.append([
                            "Polymarket",
                            token.get('token_id', 'N/A')[:10] + "...",
                            market.get('question', 'N/A')[:50] + "..." if len(market.get('question', '')) > 50 else market.get('question', 'N/A'),
                            "N/A",  # Would need to fetch price
                            "N/A"   # Would need to fetch volume
                        ])
            
            if markets_data:
                headers = ["Platform", "ID", "Title", "Price", "Volume"]
                click.echo(f"\n{Fore.CYAN}Available Markets:{Style.RESET_ALL}")
                click.echo(tabulate(markets_data, headers=headers, tablefmt="grid"))
            else:
                click.echo(f"{Fore.YELLOW}No markets found{Style.RESET_ALL}")
                
        except Exception as e:
            click.echo(f"{Fore.RED}✗ Error fetching markets: {e}{Style.RESET_ALL}")
    
    asyncio.run(_list_markets())

@cli.command()
@click.option('--config', default='bot_config.yaml', help='Configuration file path')
@click.option('--dry-run', is_flag=True, help='Run in simulation mode without placing orders')
def find_arbitrage(config, dry_run):
    """Find arbitrage opportunities"""
    async def _find_arbitrage():
        try:
            config_obj = load_config(config)
            
            # Initialize clients
            kalshi_client = KalshiClient(
                email=config_obj.kalshi_email,
                password=config_obj.kalshi_password,
                demo=config_obj.kalshi_demo
            )
            
            polymarket_client = PolymarketClient(
                private_key=config_obj.polymarket_private_key,
                funder_address=config_obj.polymarket_funder,
                demo=config_obj.polymarket_demo
            )
            
            # Initialize bot
            bot = MarketMakingBot(kalshi_client, polymarket_client)
            await bot.initialize()
            
            click.echo(f"{Fore.CYAN}Scanning for arbitrage opportunities...{Style.RESET_ALL}")
            
            # Fetch market data
            markets = await bot.fetch_all_market_data()
            click.echo(f"Fetched {len(markets)} markets")
            
            # Find similar markets
            similar_pairs = bot.find_similar_markets(markets)
            click.echo(f"Found {len(similar_pairs)} similar market pairs")
            
            # Find arbitrage opportunities
            opportunities = bot.identify_arbitrage_opportunities(similar_pairs)
            
            if opportunities:
                arb_data = []
                for opp in opportunities[:10]:  # Show top 10
                    arb_data.append([
                        opp.buy_platform.upper(),
                        opp.sell_platform.upper(),
                        f"{opp.profit_margin:.1%}",
                        f"${opp.max_volume:.0f}",
                        f"{opp.buy_price:.3f}",
                        f"{opp.sell_price:.3f}"
                    ])
                
                headers = ["Buy Platform", "Sell Platform", "Profit", "Max Volume", "Buy Price", "Sell Price"]
                click.echo(f"\n{Fore.GREEN}Arbitrage Opportunities Found:{Style.RESET_ALL}")
                click.echo(tabulate(arb_data, headers=headers, tablefmt="grid"))
                
                if dry_run:
                    click.echo(f"\n{Fore.YELLOW}DRY RUN: No orders will be placed{Style.RESET_ALL}")
                else:
                    click.echo(f"\n{Fore.RED}LIVE MODE: Orders would be placed!{Style.RESET_ALL}")
                    
            else:
                click.echo(f"{Fore.YELLOW}No arbitrage opportunities found{Style.RESET_ALL}")
                
        except Exception as e:
            click.echo(f"{Fore.RED}✗ Error finding arbitrage: {e}{Style.RESET_ALL}")
    
    asyncio.run(_find_arbitrage())

@cli.command()
@click.option('--config', default='bot_config.yaml', help='Configuration file path')
@click.option('--strategy', type=click.Choice(['arbitrage', 'market_making']), default='arbitrage')
@click.option('--duration', default=3600, help='Run duration in seconds')
@click.option('--dry-run', is_flag=True, help='Run in simulation mode')
def run_bot(config, strategy, duration, dry_run):
    """Run the trading bot"""
    async def _run_bot():
        try:
            config_obj = load_config(config)
            
            if dry_run:
                click.echo(f"{Fore.YELLOW}🔍 SIMULATION MODE - No real orders will be placed{Style.RESET_ALL}")
            else:
                click.echo(f"{Fore.RED}⚠️  LIVE TRADING MODE - Real orders will be placed!{Style.RESET_ALL}")
                if not click.confirm("Are you sure you want to continue?"):
                    return
            
            # Initialize clients
            kalshi_client = KalshiClient(
                email=config_obj.kalshi_email,
                password=config_obj.kalshi_password,
                demo=config_obj.kalshi_demo or dry_run
            )
            
            polymarket_client = PolymarketClient(
                private_key=config_obj.polymarket_private_key,
                funder_address=config_obj.polymarket_funder,
                demo=config_obj.polymarket_demo or dry_run
            )
            
            # Initialize bot
            if strategy == 'arbitrage':
                bot = MarketMakingBot(kalshi_client, polymarket_client)
                await bot.initialize()
                
                click.echo(f"{Fore.GREEN}🚀 Starting arbitrage bot for {duration} seconds...{Style.RESET_ALL}")
                
                bot.start()
                
                # Run for specified duration
                start_time = datetime.now()
                try:
                    await asyncio.wait_for(bot.run_arbitrage_strategy(), timeout=duration)
                except asyncio.TimeoutError:
                    click.echo(f"{Fore.CYAN}⏰ Bot stopped after {duration} seconds{Style.RESET_ALL}")
                
                bot.stop()
                
            elif strategy == 'market_making':
                # Advanced market making strategy
                advanced_bot = AdvancedMarketMaker(config_obj)
                click.echo(f"{Fore.GREEN}🚀 Starting market making bot...{Style.RESET_ALL}")
                click.echo(f"{Fore.YELLOW}Market making strategy not fully implemented yet{Style.RESET_ALL}")
            
        except KeyboardInterrupt:
            click.echo(f"\n{Fore.YELLOW}Bot stopped by user{Style.RESET_ALL}")
        except Exception as e:
            click.echo(f"{Fore.RED}✗ Error running bot: {e}{Style.RESET_ALL}")
    
    asyncio.run(_run_bot())

@cli.command()
@click.option('--config', default='bot_config.yaml', help='Configuration file path')
def test_connection(config):
    """Test API connections"""
    async def _test_connection():
        try:
            config_obj = load_config(config)
            
            click.echo(f"{Fore.CYAN}Testing API connections...{Style.RESET_ALL}")
            
            # Test Kalshi
            try:
                kalshi_client = KalshiClient(
                    email=config_obj.kalshi_email,
                    password=config_obj.kalshi_password,
                    demo=config_obj.kalshi_demo
                )
                await kalshi_client.authenticate()
                click.echo(f"{Fore.GREEN}✓ Kalshi connection successful{Style.RESET_ALL}")
            except Exception as e:
                click.echo(f"{Fore.RED}✗ Kalshi connection failed: {e}{Style.RESET_ALL}")
            
            # Test Polymarket
            try:
                polymarket_client = PolymarketClient(
                    private_key=config_obj.polymarket_private_key,
                    funder_address=config_obj.polymarket_funder,
                    demo=config_obj.polymarket_demo
                )
                await polymarket_client.initialize()
                click.echo(f"{Fore.GREEN}✓ Polymarket connection successful{Style.RESET_ALL}")
            except Exception as e:
                click.echo(f"{Fore.RED}✗ Polymarket connection failed: {e}{Style.RESET_ALL}")
                
        except Exception as e:
            click.echo(f"{Fore.RED}✗ Error testing connections: {e}{Style.RESET_ALL}")
    
    asyncio.run(_test_connection())

@cli.command()
def status():
    """Show bot status and recent activity"""
    try:
        # Check if log file exists
        log_file = Path("bot.log")
        if log_file.exists():
            click.echo(f"{Fore.CYAN}Recent Activity:{Style.RESET_ALL}")
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-10:]:  # Show last 10 lines
                    click.echo(line.strip())
        else:
            click.echo(f"{Fore.YELLOW}No activity log found{Style.RESET_ALL}")
            
        # Check if config exists
        config_file = Path("bot_config.yaml")
        if config_file.exists():
            click.echo(f"\n{Fore.GREEN}✓ Configuration file found{Style.RESET_ALL}")
        else:
            click.echo(f"\n{Fore.RED}✗ Configuration file missing{Style.RESET_ALL}")
            click.echo(f"Run 'python cli.py init-config' to create one")
            
    except Exception as e:
        click.echo(f"{Fore.RED}✗ Error checking status: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    cli()

# Example usage commands:
"""
# Initialize configuration
python cli.py init-config

# Validate configuration
python cli.py validate-config

# Test API connections
python cli.py test-connection

# List markets
python cli.py list-markets --platform both --limit 20

# Find arbitrage opportunities (simulation)
python cli.py find-arbitrage --dry-run

# Run arbitrage bot for 1 hour (simulation)
python cli.py run-bot --strategy arbitrage --duration 3600 --dry-run

# Run live arbitrage bot (REAL MONEY!)
python cli.py run-bot --strategy arbitrage --duration 3600

# Check bot status
python cli.py status
"""
