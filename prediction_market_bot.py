import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import aiohttp
import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Unified market data structure"""
    platform: str
    market_id: str
    title: str
    yes_price: float
    no_price: float
    yes_volume: float
    no_volume: float
    bid_yes: Optional[float] = None
    ask_yes: Optional[float] = None
    bid_no: Optional[float] = None
    ask_no: Optional[float] = None
    timestamp: float = None

@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity between platforms"""
    buy_platform: str
    sell_platform: str
    buy_market_id: str
    sell_market_id: str
    buy_side: str  # 'yes' or 'no'
    sell_side: str  # 'yes' or 'no'
    buy_price: float
    sell_price: float
    profit_margin: float
    max_volume: float

class KalshiClient:
    """Kalshi API client wrapper"""
    
    def __init__(self, email: str, password: str, demo: bool = True):
        self.email = email
        self.password = password
        self.base_url = "https://demo-api.kalshi.co/trade-api/v2" if demo else "https://api.elections.kalshi.com/trade-api/v2"
        self.token = None
        self.headers = {"accept": "application/json", "content-type": "application/json"}
        
    async def authenticate(self):
        """Authenticate and get bearer token"""
        url = f"{self.base_url}/login"
        body = {"email": self.email, "password": self.password}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=body) as response:
                data = await response.json()
                self.token = f"Bearer {data['token']}"
                self.headers["Authorization"] = self.token
                logger.info("Kalshi authentication successful")
    
    async def get_markets(self, status: str = "open", limit: int = 100) -> List[Dict]:
        """Get markets from Kalshi"""
        url = f"{self.base_url}/markets?status={status}&limit={limit}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                data = await response.json()
                return data.get('markets', [])
    
    async def get_orderbook(self, market_ticker: str) -> Dict:
        """Get orderbook for a specific market"""
        url = f"{self.base_url}/markets/{market_ticker}/orderbook"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                return await response.json()
    
    async def place_order(self, market_ticker: str, side: str, price: float, quantity: int) -> Dict:
        """Place an order on Kalshi"""
        url = f"{self.base_url}/orders"
        
        order_data = {
            "ticker": market_ticker,
            "client_order_id": f"bot_{int(time.time())}",
            "side": side,  # "yes" or "no"
            "action": "buy",  # or "sell"
            "count": quantity,
            "type": "limit",
            "yes_price": int(price * 100) if side == "yes" else None,
            "no_price": int(price * 100) if side == "no" else None
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=order_data) as response:
                return await response.json()

class PolymarketClient:
    """Polymarket client wrapper using py-clob-client"""
    
    def __init__(self, private_key: str, funder_address: str, demo: bool = True):
        self.host = "https://clob.polymarket.com"
        self.chain_id = 137  # Polygon mainnet
        self.private_key = private_key
        self.funder = funder_address
        
        self.client = ClobClient(
            host=self.host,
            key=private_key,
            chain_id=self.chain_id,
            signature_type=1,
            funder=funder_address
        )
        
    async def initialize(self):
        """Initialize the Polymarket client"""
        try:
            self.client.set_api_creds(self.client.create_or_derive_api_creds())
            logger.info("Polymarket client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Polymarket client: {e}")
            raise
    
    async def get_markets(self) -> List[Dict]:
        """Get markets from Polymarket Gamma API"""
        url = "https://gamma-api.polymarket.com/markets"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                return data if isinstance(data, list) else data.get('markets', [])
    
    def get_orderbook(self, token_id: str) -> Dict:
        """Get orderbook for a token"""
        return self.client.get_order_book(token_id)
    
    def get_midpoint(self, token_id: str) -> float:
        """Get midpoint price for a token"""
        return self.client.get_midpoint(token_id)
    
    def place_order(self, token_id: str, price: float, size: float, side: str) -> Dict:
        """Place an order on Polymarket"""
        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=BUY if side.lower() == 'buy' else SELL
        )
        
        signed_order = self.client.create_order(order_args)
        return self.client.post_order(signed_order, OrderType.GTC)

class MarketMakingBot:
    """Main market making bot class"""
    
    def __init__(self, kalshi_client: KalshiClient, polymarket_client: PolymarketClient):
        self.kalshi = kalshi_client
        self.polymarket = polymarket_client
        self.market_cache = {}
        self.running = False
        
        # Bot configuration
        self.min_profit_margin = 0.02  # 2% minimum profit
        self.max_position_size = 100   # Maximum position size
        self.update_interval = 10      # Seconds between updates
        
    async def initialize(self):
        """Initialize both clients"""
        await self.kalshi.authenticate()
        await self.polymarket.initialize()
        logger.info("Market making bot initialized")
    
    def convert_kalshi_to_market_data(self, market: Dict) -> MarketData:
        """Convert Kalshi market data to unified format"""
        return MarketData(
            platform="kalshi",
            market_id=market.get('ticker', ''),
            title=market.get('title', ''),
            yes_price=market.get('yes_price', 0) / 100.0,  # Kalshi prices in cents
            no_price=market.get('no_price', 0) / 100.0,
            yes_volume=market.get('volume', 0),
            no_volume=market.get('volume', 0),  # Kalshi doesn't separate yes/no volume
            timestamp=time.time()
        )
    
    def convert_polymarket_to_market_data(self, market: Dict) -> List[MarketData]:
        """Convert Polymarket market data to unified format"""
        market_data_list = []
        
        # Polymarket markets can have multiple outcomes (tokens)
        tokens = market.get('tokens', [])
        for token in tokens:
            try:
                # Get current price data
                token_id = token.get('token_id', '')
                if not token_id:
                    continue
                    
                midpoint = self.polymarket.get_midpoint(token_id)
                orderbook = self.polymarket.get_orderbook(token_id)
                
                market_data = MarketData(
                    platform="polymarket",
                    market_id=token_id,
                    title=f"{market.get('question', '')} - {token.get('outcome', '')}",
                    yes_price=midpoint,
                    no_price=1.0 - midpoint,
                    yes_volume=0,  # Would need to calculate from orderbook
                    no_volume=0,
                    timestamp=time.time()
                )
                
                # Add orderbook data if available
                if orderbook and 'bids' in orderbook:
                    bids = orderbook.get('bids', [])
                    asks = orderbook.get('asks', [])
                    if bids:
                        market_data.bid_yes = float(bids[0]['price'])
                    if asks:
                        market_data.ask_yes = float(asks[0]['price'])
                
                market_data_list.append(market_data)
                
            except Exception as e:
                logger.warning(f"Error processing Polymarket token {token_id}: {e}")
                continue
                
        return market_data_list
    
    async def fetch_all_market_data(self) -> List[MarketData]:
        """Fetch market data from both platforms"""
        all_markets = []
        
        try:
            # Fetch Kalshi markets
            kalshi_markets = await self.kalshi.get_markets()
            for market in kalshi_markets:
                market_data = self.convert_kalshi_to_market_data(market)
                all_markets.append(market_data)
                
        except Exception as e:
            logger.error(f"Error fetching Kalshi markets: {e}")
        
        try:
            # Fetch Polymarket markets
            polymarket_markets = await self.polymarket.get_markets()
            for market in polymarket_markets[:10]:  # Limit for testing
                market_data_list = self.convert_polymarket_to_market_data(market)
                all_markets.extend(market_data_list)
                
        except Exception as e:
            logger.error(f"Error fetching Polymarket markets: {e}")
        
        return all_markets
    
    def find_similar_markets(self, markets: List[MarketData]) -> List[Tuple[MarketData, MarketData]]:
        """Find potentially similar markets across platforms"""
        similar_pairs = []
        
        kalshi_markets = [m for m in markets if m.platform == "kalshi"]
        polymarket_markets = [m for m in markets if m.platform == "polymarket"]
        
        for k_market in kalshi_markets:
            for p_market in polymarket_markets:
                # Simple similarity check based on keywords
                k_words = set(k_market.title.lower().split())
                p_words = set(p_market.title.lower().split())
                
                # Calculate word overlap
                overlap = len(k_words.intersection(p_words))
                if overlap >= 2:  # At least 2 common words
                    similar_pairs.append((k_market, p_market))
        
        return similar_pairs
    
    def identify_arbitrage_opportunities(self, market_pairs: List[Tuple[MarketData, MarketData]]) -> List[ArbitrageOpportunity]:
        """Identify arbitrage opportunities between similar markets"""
        opportunities = []
        
        for market1, market2 in market_pairs:
            # Check for arbitrage in both directions
            
            # Buy on market1, sell on market2
            if market1.yes_price < market2.yes_price:
                profit_margin = market2.yes_price - market1.yes_price
                if profit_margin > self.min_profit_margin:
                    opportunities.append(ArbitrageOpportunity(
                        buy_platform=market1.platform,
                        sell_platform=market2.platform,
                        buy_market_id=market1.market_id,
                        sell_market_id=market2.market_id,
                        buy_side="yes",
                        sell_side="yes",
                        buy_price=market1.yes_price,
                        sell_price=market2.yes_price,
                        profit_margin=profit_margin,
                        max_volume=min(market1.yes_volume, market2.yes_volume)
                    ))
            
            # Buy on market2, sell on market1
            if market2.yes_price < market1.yes_price:
                profit_margin = market1.yes_price - market2.yes_price
                if profit_margin > self.min_profit_margin:
                    opportunities.append(ArbitrageOpportunity(
                        buy_platform=market2.platform,
                        sell_platform=market1.platform,
                        buy_market_id=market2.market_id,
                        sell_market_id=market1.market_id,
                        buy_side="yes",
                        sell_side="yes",
                        buy_price=market2.yes_price,
                        sell_price=market1.yes_price,
                        profit_margin=profit_margin,
                        max_volume=min(market1.yes_volume, market2.yes_volume)
                    ))
        
        return opportunities
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity):
        """Execute an arbitrage trade"""
        logger.info(f"Executing arbitrage: Buy {opportunity.buy_side} on {opportunity.buy_platform} "
                   f"at {opportunity.buy_price}, Sell on {opportunity.sell_platform} "
                   f"at {opportunity.sell_price}, Profit: {opportunity.profit_margin:.3f}")
        
        try:
            # Calculate position size (conservative approach)
            position_size = min(self.max_position_size, opportunity.max_volume * 0.1)
            
            # Execute buy order
            if opportunity.buy_platform == "kalshi":
                buy_result = await self.kalshi.place_order(
                    opportunity.buy_market_id,
                    opportunity.buy_side,
                    opportunity.buy_price,
                    int(position_size)
                )
            else:  # polymarket
                buy_result = self.polymarket.place_order(
                    opportunity.buy_market_id,
                    opportunity.buy_price,
                    position_size,
                    "buy"
                )
            
            logger.info(f"Buy order result: {buy_result}")
            
            # Execute sell order (in a real implementation, you'd want to confirm the buy first)
            if opportunity.sell_platform == "kalshi":
                sell_result = await self.kalshi.place_order(
                    opportunity.sell_market_id,
                    opportunity.sell_side,
                    opportunity.sell_price,
                    int(position_size)
                )
            else:  # polymarket
                sell_result = self.polymarket.place_order(
                    opportunity.sell_market_id,
                    opportunity.sell_price,
                    position_size,
                    "sell"
                )
            
            logger.info(f"Sell order result: {sell_result}")
            
        except Exception as e:
            logger.error(f"Error executing arbitrage: {e}")
    
    async def run_arbitrage_strategy(self):
        """Main arbitrage strategy loop"""
        logger.info("Starting arbitrage strategy")
        
        while self.running:
            try:
                # Fetch market data
                markets = await self.fetch_all_market_data()
                logger.info(f"Fetched {len(markets)} markets")
                
                # Find similar markets
                similar_pairs = self.find_similar_markets(markets)
                logger.info(f"Found {len(similar_pairs)} similar market pairs")
                
                # Identify arbitrage opportunities
                opportunities = self.identify_arbitrage_opportunities(similar_pairs)
                logger.info(f"Found {len(opportunities)} arbitrage opportunities")
                
                # Execute profitable trades
                for opportunity in opportunities[:3]:  # Limit to top 3 opportunities
                    await self.execute_arbitrage(opportunity)
                
                # Wait before next iteration
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in arbitrage strategy loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    def start(self):
        """Start the bot"""
        self.running = True
        
    def stop(self):
        """Stop the bot"""
        self.running = False

# Example usage
async def main():
    """Example main function"""
    
    # Initialize clients (replace with your actual credentials)
    kalshi_client = KalshiClient(
        email="your_email@example.com",
        password="your_password",
        demo=True
    )
    
    polymarket_client = PolymarketClient(
        private_key="your_private_key",
        funder_address="your_funder_address",
        demo=True
    )
    
    # Create bot
    bot = MarketMakingBot(kalshi_client, polymarket_client)
    
    # Initialize
    await bot.initialize()
    
    # Start arbitrage strategy
    bot.start()
    
    try:
        await bot.run_arbitrage_strategy()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    finally:
        bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
