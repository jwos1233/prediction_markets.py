import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class BotConfig:
    """Bot configuration settings"""
    # Trading parameters
    min_profit_margin: float = 0.02
    max_position_size: float = 100.0
    update_interval: int = 10
    
    # Risk management
    max_daily_loss: float = 500.0
    max_positions: int = 10
    position_timeout: int = 3600  # 1 hour
    
    # Market making parameters
    spread_width: float = 0.01  # 1% spread
    inventory_target: float = 0.0  # Neutral inventory target
    inventory_penalty: float = 0.001  # Penalty per unit of inventory
    
    # Kalshi settings
    kalshi_email: str = ""
    kalshi_password: str = ""
    kalshi_demo: bool = True
    
    # Polymarket settings
    polymarket_private_key: str = ""
    polymarket_funder: str = ""
    polymarket_demo: bool = True
    
    # Market matching
    title_similarity_threshold: float = 0.6
    price_correlation_threshold: float = 0.7
    min_volume_threshold: float = 10.0

class PricingModel:
    """Advanced pricing models for prediction markets"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.price_history = {}
        self.correlation_cache = {}
        
    def calculate_fair_value(self, market_data: List[Dict]) -> float:
        """Calculate fair value using multiple pricing methods"""
        if not market_data:
            return 0.5  # Default neutral price
        
        # Method 1: Volume-weighted average price
        vwap = self.calculate_vwap(market_data)
        
        # Method 2: Bid-ask midpoint
        midpoint = self.calculate_midpoint(market_data)
        
        # Method 3: Historical trend analysis
        trend_price = self.calculate_trend_price(market_data)
        
        # Method 4: Cross-platform consensus
        consensus_price = self.calculate_consensus_price(market_data)
        
        # Weighted combination of methods
        weights = [0.3, 0.3, 0.2, 0.2]
        prices = [vwap, midpoint, trend_price, consensus_price]
        
        # Filter out None values
        valid_prices = [(w, p) for w, p in zip(weights, prices) if p is not None]
        
        if not valid_prices:
            return 0.5
        
        total_weight = sum(w for w, p in valid_prices)
        weighted_price = sum(w * p for w, p in valid_prices) / total_weight
        
        return max(0.01, min(0.99, weighted_price))
    
    def calculate_vwap(self, market_data: List[Dict]) -> Optional[float]:
        """Calculate volume-weighted average price"""
        try:
            total_volume = 0
            total_value = 0
            
            for data in market_data:
                volume = data.get('volume', 0)
                price = data.get('price', 0)
                
                if volume > 0 and price > 0:
                    total_volume += volume
                    total_value += volume * price
            
            return total_value / total_volume if total_volume > 0 else None
            
        except Exception as e:
            logger.warning(f"Error calculating VWAP: {e}")
            return None
    
    def calculate_midpoint(self, market_data: List[Dict]) -> Optional[float]:
        """Calculate bid-ask midpoint"""
        try:
            bids = []
            asks = []
            
            for data in market_data:
                if 'bid' in data and data['bid'] > 0:
                    bids.append(data['bid'])
                if 'ask' in data and data['ask'] > 0:
                    asks.append(data['ask'])
            
            if bids and asks:
                best_bid = max(bids)
                best_ask = min(asks)
                return (best_bid + best_ask) / 2
            
            return None
            
        except Exception as e:
            logger.warning(f"Error calculating midpoint: {e}")
            return None
    
    def calculate_trend_price(self, market_data: List[Dict]) -> Optional[float]:
        """Calculate price based on recent trends"""
        try:
            if len(market_data) < 3:
                return None
            
            # Extract recent prices
            prices = []
            timestamps = []
            
            for data in market_data:
                if 'price' in data and 'timestamp' in data:
                    prices.append(data['price'])
                    timestamps.append(data['timestamp'])
            
            if len(prices) < 3:
                return None
            
            # Fit linear trend
            x = np.array(timestamps)
            y = np.array(prices)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Project to current time
            current_time = datetime.now().timestamp()
            trend_price = slope * current_time + intercept
            
            # Weight by R-squared (confidence in trend)
            confidence = r_value ** 2
            recent_price = prices[-1]
            
            # Blend trend prediction with recent price
            return confidence * trend_price + (1 - confidence) * recent_price
            
        except Exception as e:
            logger.warning(f"Error calculating trend price: {e}")
            return None
    
    def calculate_consensus_price(self, market_data: List[Dict]) -> Optional[float]:
        """Calculate consensus price across platforms"""
        try:
            platform_prices = {}
            
            for data in market_data:
                platform = data.get('platform')
                price = data.get('price')
                
                if platform and price is not None:
                    if platform not in platform_prices:
                        platform_prices[platform] = []
                    platform_prices[platform].append(price)
            
            if len(platform_prices) < 2:
                return None
            
            # Calculate average price per platform
            platform_averages = {}
            for platform, prices in platform_prices.items():
                platform_averages[platform] = np.mean(prices)
            
            # Return median of platform averages
            return np.median(list(platform_averages.values()))
            
        except Exception as e:
            logger.warning(f"Error calculating consensus price: {e}")
            return None
    
    def calculate_volatility(self, price_history: List[float], window: int = 20) -> float:
        """Calculate price volatility"""
        if len(price_history) < window:
            return 0.1  # Default volatility
        
        recent_prices = price_history[-window:]
        returns = np.diff(np.log(recent_prices))
        return np.std(returns) * np.sqrt(252)  # Annualized volatility

class MarketMatcher:
    """Advanced market matching across platforms"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.similarity_cache = {}
        
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using Jaccard similarity"""
        # Normalize text
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Create word sets
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'will', 'be', 'is', 'are', 'was', 'were'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_price_correlation(self, prices1: List[float], prices2: List[float]) -> float:
        """Calculate price correlation between markets"""
        if len(prices1) < 3 or len(prices2) < 3:
            return 0.0
        
        try:
            correlation, p_value = stats.pearsonr(prices1, prices2)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def find_matching_markets(self, kalshi_markets: List[Dict], polymarket_markets: List[Dict]) -> List[Tuple[Dict, Dict, float]]:
        """Find matching markets across platforms with confidence scores"""
        matches = []
        
        for k_market in kalshi_markets:
            for p_market in polymarket_markets:
                # Calculate text similarity
                text_sim = self.calculate_text_similarity(
                    k_market.get('title', ''),
                    p_market.get('question', '')
                )
                
                if text_sim < self.config.title_similarity_threshold:
                    continue
                
                # Calculate additional features for matching
                features = {
                    'text_similarity': text_sim,
                    'price_diff': abs(k_market.get('yes_price', 0.5) - p_market.get('price', 0.5)),
                    'volume_ratio': min(k_market.get('volume', 1), p_market.get('volume', 1)) / max(k_market.get('volume', 1), p_market.get('volume', 1)),
                }
                
                # Combined confidence score
                confidence = (
                    features['text_similarity'] * 0.6 +
                    (1 - features['price_diff']) * 0.3 +
                    features['volume_ratio'] * 0.1
                )
                
                if confidence > 0.7:  # High confidence threshold
                    matches.append((k_market, p_market, confidence))
        
        # Sort by confidence
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches

class RiskManager:
    """Risk management for the trading bot"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.positions = {}
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()
        
    def check_risk_limits(self, proposed_trade: Dict) -> Tuple[bool, str]:
        """Check if proposed trade meets risk limits"""
        
        # Check daily loss limit
        if self.daily_pnl < -self.config.max_daily_loss:
            return False, "Daily loss limit exceeded"
        
        # Check position count
        if len(self.positions) >= self.config.max_positions:
            return False, "Maximum position count reached"
        
        # Check position size
        if proposed_trade.get('size', 0) > self.config.max_position_size:
            return False, "Position size too large"
        
        # Check minimum profit margin
        if proposed_trade.get('profit_margin', 0) < self.config.min_profit_margin:
            return False, "Profit margin too low"
        
        return True, "Risk checks passed"
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L"""
        current_date = datetime.now().date()
        
        # Reset daily P&L if new day
        if current_date != self.last_reset:
            self.daily_pnl = 0.0
            self.last_reset = current_date
        
        self.daily_pnl += pnl
    
    def add_position(self, position_id: str, position_data: Dict):
        """Add a new position"""
        self.positions[position_id] = {
            **position_data,
            'timestamp': datetime.now(),
            'status': 'open'
        }
    
    def close_position(self, position_id: str, pnl: float):
        """Close a position and update P&L"""
        if position_id in self.positions:
            self.positions[position_id]['status'] = 'closed'
            self.positions[position_id]['pnl'] = pnl
            self.update_daily_pnl(pnl)
    
    def check_position_timeouts(self) -> List[str]:
        """Check for positions that should be closed due to timeout"""
        timeout_positions = []
        current_time = datetime.now()
        
        for pos_id, position in self.positions.items():
            if position['status'] == 'open':
                time_diff = (current_time - position['timestamp']).seconds
                if time_diff > self.config.position_timeout:
                    timeout_positions.append(pos_id)
        
        return timeout_positions

class AdvancedMarketMaker:
    """Advanced market making strategy"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.pricing_model = PricingModel(config)
        self.market_matcher = MarketMatcher(config)
        self.risk_manager = RiskManager(config)
        self.inventory = {}  # Track inventory by market
        
    def calculate_optimal_quotes(self, market_data: Dict) -> Tuple[float, float]:
        """Calculate optimal bid and ask quotes"""
        
        # Get fair value
        fair_value = self.pricing_model.calculate_fair_value([market_data])
        
        # Base spread
        base_spread = self.config.spread_width
        
        # Adjust spread based on volatility
        price_history = self.pricing_model.price_history.get(market_data.get('id'), [])
        volatility = self.pricing_model.calculate_volatility(price_history)
        volatility_adjustment = volatility * 0.5
        
        # Adjust for inventory
        inventory_position = self.inventory.get(market_data.get('id'), 0)
        inventory_adjustment = inventory_position * self.config.inventory_penalty
        
        # Calculate half-spread
        half_spread = (base_spread + volatility_adjustment) / 2
        
        # Apply inventory adjustment
        bid = fair_value - half_spread - inventory_adjustment
        ask = fair_value + half_spread + inventory_adjustment
        
        # Ensure valid price range
        bid = max(0.01, min(0.99, bid))
        ask = max(0.01, min(0.99, ask))
        
        # Ensure bid < ask
        if bid >= ask:
            mid = (bid + ask) / 2
            bid = mid - 0.005
            ask = mid + 0.005
        
        return bid, ask
    
    def generate_trading_signals(self, matched_markets: List[Tuple[Dict, Dict, float]]) -> List[Dict]:
        """Generate trading signals from matched markets"""
        signals = []
        
        for kalshi_market, poly_market, confidence in matched_markets:
            # Calculate fair values
            kalshi_fair = self.pricing_model.calculate_fair_value([kalshi_market])
            poly_fair = self.pricing_model.calculate_fair_value([poly_market])
            
            # Check for arbitrage opportunities
            price_diff = abs(kalshi_fair - poly_fair)
            
            if price_diff > self.config.min_profit_margin:
                # Determine trade direction
                if kalshi_fair < poly_fair:
                    # Buy Kalshi, sell Polymarket
                    signal = {
                        'type': 'arbitrage',
                        'buy_platform': 'kalshi',
                        'sell_platform': 'polymarket',
                        'buy_market': kalshi_market,
                        'sell_market': poly_market,
                        'profit_margin': price_diff,
                        'confidence': confidence,
                        'fair_value_diff': price_diff
                    }
                else:
                    # Buy Polymarket, sell Kalshi
                    signal = {
                        'type': 'arbitrage',
                        'buy_platform': 'polymarket',
                        'sell_platform': 'kalshi',
                        'buy_market': poly_market,
                        'sell_market': kalshi_market,
                        'profit_margin': price_diff,
                        'confidence': confidence,
                        'fair_value_diff': price_diff
                    }
                
                signals.append(signal)
        
        # Sort by profit potential
        signals.sort(key=lambda x: x['profit_margin'] * x['confidence'], reverse=True)
        
        return signals

# Configuration loader
def load_config(config_file: str = "bot_config.yaml") -> BotConfig:
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        return BotConfig(**config_dict)
    except FileNotFoundError:
        logger.warning(f"Config file {config_file} not found, using defaults")
        return BotConfig()
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return BotConfig()

# Example configuration file content
EXAMPLE_CONFIG = """
# Trading parameters
min_profit_margin: 0.02
max_position_size: 100.0
update_interval: 10

# Risk management
max_daily_loss: 500.0
max_positions: 10
position_timeout: 3600

# Market making parameters
spread_width: 0.01
inventory_target: 0.0
inventory_penalty: 0.001

# Kalshi settings
kalshi_email: "your_email@example.com"
kalshi_password: "your_password"
kalshi_demo: true

# Polymarket settings
polymarket_private_key: "your_private_key"
polymarket_funder: "your_funder_address"
polymarket_demo: true

# Market matching
title_similarity_threshold: 0.6
price_correlation_threshold: 0.7
min_volume_threshold: 10.0
"""

def create_example_config():
    """Create an example configuration file"""
    with open("bot_config.yaml", "w") as f:
        f.write(EXAMPLE_CONFIG)
    print("Created example configuration file: bot_config.yaml")

if __name__ == "__main__":
    create_example_config()
