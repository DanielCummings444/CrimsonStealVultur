"""
   Before running this script, ensure you have the required packages installed:
   pip install -r requirements.txt

"""

import os, json, math, random, logging, asyncio, pickle, requests, re
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
from logging.handlers import RotatingFileHandler
from collections import namedtuple
from urllib.parse import urlencode

import nest_asyncio
nest_asyncio.apply()

# Alpaca SDK
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, APIError
from alpaca_trade_api.stream import Stream

# PyTorch for RL
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F

# TA-Lib indicators
import ta
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator

# HuggingFace Transformers
from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis", 
                              model="distilbert-base-uncased-finetuned-sst-2-english", 
                              revision="main", 
                              framework="pt")

###############################################################################
# Cache System for Historical Bars
###############################################################################
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_filepath(symbol: str, timeframe: str, days: int) -> str:
    filename = f"{symbol}_{timeframe}_{days}d.pkl"
    return os.path.join(CACHE_DIR, filename)

def load_cached_bars(symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
    filepath = get_cache_filepath(symbol, timeframe, days)
    if os.path.exists(filepath):
        try:
            return pd.read_pickle(filepath)
        except Exception as e:
            logger.error(f"Failed to load cache for {symbol}: {e}", exc_info=True)
    return None

def save_cached_bars(symbol: str, timeframe: str, days: int, df: pd.DataFrame):
    filepath = get_cache_filepath(symbol, timeframe, days)
    try:
        df.to_pickle(filepath)
        logger.info(f"Cached bars saved for {symbol} at {filepath}")
    except Exception as e:
        logger.error(f"Failed to save cache for {symbol}: {e}", exc_info=True)

###############################################################################
# Helper Functions: Market Hours
###############################################################################
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from pytz import timezone as ZoneInfo

def is_market_hours() -> bool:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close

def time_until_market_open() -> float:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_et < market_open:
        return (market_open - now_et).total_seconds()
    else:
        next_day = now_et + timedelta(days=1)
        market_open_next = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        return (market_open_next - now_et).total_seconds()

###############################################################################
# 1) Load Config and Setup Logging
###############################################################################
CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("config.json not found")

def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

config = load_config()
API_KEY = config.get("API_KEY")
API_SECRET = config.get("API_SECRET")
PAPER = config.get("PAPER", True)
BASE_URL = config.get("BASE_URL", "https://paper-api.alpaca.markets")
TRADE_OPTIONS = config.get("trade_options", True)
OPTIONS_DATA_URL = config.get("OPTIONS_DATA_URL", "https://data.alpaca.markets")

LOG_FILE = config.get("LOG_FILE", "trading_bot.log")
LOG_LEVEL_STR = config.get("LOG_LEVEL", "INFO").upper()

MEMORY_SIZE = config.get("MEMORY_SIZE", 100000)
BATCH_SIZE = config.get("BATCH_SIZE", 64)
GAMMA = config.get("GAMMA", 0.99)
LEARNING_RATE = config.get("LEARNING_RATE", 0.0005)
TARGET_UPDATE = config.get("TARGET_UPDATE", 1000)
EPSILON_DECAY = config.get("EPSILON_DECAY", 10000)
ALPHA = config.get("ALPHA", 0.6)
BETA_START = config.get("BETA_START", 0.4)

RISK_PER_TRADE = config.get("RISK_PER_TRADE", 0.02)
ATR_PERIOD = config.get("ATR_PERIOD", 14)
VOL_TRAILING_MULTIPLIER = config.get("VOL_TRAILING_MULTIPLIER", 1.5)
MAX_DRAWDOWN = config.get("MAX_DRAWDOWN_LIMIT", 0.2)
TRAILING_STOP_LOSS = config.get("TRAILING_STOP_LOSS", 0.02)
TAKE_PROFIT = config.get("TAKE_PROFIT", 0.03)
MIN_HOLD_DAYS = config.get("MIN_HOLD_DAYS", 1)
MAX_HOLD_DAYS = config.get("MAX_HOLD_DAYS", 10)

TRANSFORMER_LAYERS = config.get("TRANSFORMER_LAYERS", 3)
ATTENTION_HEADS = config.get("ATTENTION_HEADS", 4)

if not API_KEY or not API_SECRET:
    raise ValueError("API_KEY or API_SECRET not found in config.json")

logger = logging.getLogger("MasterpieceTrader")
logger.setLevel(getattr(logging, LOG_LEVEL_STR, logging.INFO))
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10**6, backupCount=5)
file_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)

###############################################################################
# Set device
###############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device set to use {device}")

###############################################################################
# Global containers
###############################################################################
live_bars: Dict[str, List[Dict]] = {}

###############################################################################
# 2) Alpaca Trader Class
###############################################################################
class AlpacaTrader:
    def __init__(self, config: dict, logger: logging.Logger):
        self.logger = logger
        self.rest = REST(
            key_id=config["API_KEY"],
            secret_key=config["API_SECRET"],
            base_url=config.get("BASE_URL", "https://paper-api.alpaca.markets"),
            api_version="v2"
        )
        self.initialize_market_calendar()

    def initialize_market_calendar(self):
        try:
            now = datetime.now(timezone.utc)
            date_str = now.strftime("%Y-%m-%d")
            calendars = self.rest.get_calendar(start=date_str, end=date_str)
            self.market_open = calendars[0].open if calendars else None
            self.market_close = calendars[0].close if calendars else None
            self.logger.info(f"Market Open: {self.market_open}, Market Close: {self.market_close}")
        except Exception as e:
            self.logger.error(f"Error fetching market calendar => {e}", exc_info=True)
            self.market_open = None
            self.market_close = None

    async def get_account(self):
        try:
            return self.rest.get_account()
        except Exception as e:
            self.logger.error(f"get_account => {e}", exc_info=True)
            return None

    async def submit_order(self, **kwargs):
        kwargs.pop("time_in_force", None)
        try:
            order = self.rest.submit_order(**kwargs)
            self.logger.info(f"Order Submitted => {order}")
            return order
        except Exception as e:
            self.logger.error(f"submit_order => {e}", exc_info=True)
            return None

    def get_latest_price(self, symbol: str):
        try:
            if re.match(r"^[A-Z]+[0-9]{6}[CP][0-9]{8}$", symbol):
                return self.get_latest_option_quote(symbol)
            else:
                quote = self.rest.get_latest_quote(symbol)
                if quote.bid_price is not None and quote.ask_price is not None:
                    return (quote.bid_price + quote.ask_price) / 2
                elif quote.last_price is not None:
                    return quote.last_price
                return None
        except Exception as e:
            self.logger.error(f"get_latest_price => {symbol}: {e}", exc_info=True)
            return None

    def get_latest_option_quote(self, symbol: str):
        try:
            # Extract underlying symbol from the option symbol (all leading letters)
            m = re.match(r"^([A-Z]+)", symbol)
            if not m:
                self.logger.error(f"Could not extract underlying symbol from {symbol}")
                return None
            underlying = m.group(1)
            endpoint = f"{OPTIONS_DATA_URL}/v1beta1/options/snapshots/{underlying}"
            params = {"feed": "opra", "limit": 100}
            headers = {
                "accept": "application/json",
                "APCA-API-KEY-ID": config["API_KEY"],
                "APCA-API-SECRET-KEY": config["API_SECRET"]
            }
            response = requests.get(endpoint, params=params, headers=headers)
            if response.status_code != 200:
                self.logger.error(f"Failed to get snapshots for underlying {underlying}: HTTP {response.status_code} - {response.text}")
                return None
            data = response.json()
            snapshots = data.get("snapshots", {})
            if symbol in snapshots:
                snapshot = snapshots[symbol]
                quote_data = snapshot.get("latestQuote", {})
                bid = quote_data.get("bp")
                ask = quote_data.get("ap")
                if bid is not None and ask is not None:
                    # If bid is zero (or near zero) use ask price
                    if bid <= 0:
                        return ask
                    avg_price = (bid + ask) / 2
                    self.logger.debug(f"Latest snapshot for {symbol}: bid={bid}, ask={ask}, avg={avg_price}")
                    return avg_price
                elif ask is not None:
                    return ask
            self.logger.error(f"No snapshot quote data found for {symbol}.")
            return None
        except Exception as e:
            self.logger.error(f"get_latest_option_quote => {symbol}: {e}", exc_info=True)
            return None

    def get_options_contracts(self, underlying: str, expiration_date: str) -> List[Dict]:
        url = f"{BASE_URL}/v2/options/contracts"
        params = {
            "underlying_symbols": underlying,
            "status": "active",
            "expiration_date": expiration_date,
            "limit": 1000
        }
        headers = {
            "APCA-API-KEY-ID": config["API_KEY"],
            "APCA-API-SECRET-KEY": config["API_SECRET"],
            "Accept": "application/json"
        }
        self.logger.debug(f"Fetching options contracts for {underlying} with params: {params}")
        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            self.logger.error(f"Failed to fetch options contracts: HTTP {response.status_code} - {response.text}")
            return []
        data = response.json()
        contracts = data.get("option_contracts", [])
        self.logger.info(f"Retrieved {len(contracts)} contracts for {underlying} expiring {expiration_date}")
        valid_contracts = [c for c in contracts if c.get("expiration_date") == expiration_date and c.get("tradable", False)]
        return sorted(valid_contracts, key=lambda c: float(c.get("strike_price", 0)))

###############################################################################
# 4) Historical Data Fetching and Tradable Symbols Helper
###############################################################################
def get_all_tradable_symbols(trader: AlpacaTrader, logger: logging.Logger) -> List[str]:
    try:
        assets = trader.rest.list_assets(status='active', asset_class='us_equity')
        symbols = [asset.symbol for asset in assets if asset.tradable]
        logger.info(f"Found {len(symbols)} tradable symbols.")
        return symbols
    except Exception as e:
        logger.error(f"Error retrieving assets: {e}", exc_info=True)
        return []

async def fetch_initial_bars(symbol: str, trader: AlpacaTrader, logger: logging.Logger,
                               days: int = 5, timeframe: str = "15Min", use_cache: bool = True) -> pd.DataFrame:
    if use_cache:
        cached_df = load_cached_bars(symbol, timeframe, days)
        if cached_df is not None and not cached_df.empty:
            logger.info(f"Loaded cached bars for {symbol}")
            return cached_df
    try:
        end_utc = datetime.now(timezone.utc)
        start_utc = end_utc - timedelta(days=days)
        bars = trader.rest.get_bars(symbol, timeframe, start_utc.isoformat(), end_utc.isoformat(), adjustment="raw").df
        if bars.empty:
            logger.warning(f"{symbol} => No historical bars returned.")
            return pd.DataFrame()
        bars.reset_index(inplace=True)
        bars.rename(columns={"timestamp": "timestamp", "open": "open", "high": "high",
                             "low": "low", "close": "close", "volume": "volume"}, inplace=True)
        save_cached_bars(symbol, timeframe, days, bars)
        return bars
    except Exception as e:
        logger.error(f"{symbol} => fetch_initial_bars: {e}", exc_info=True)
        return pd.DataFrame()

###############################################################################
# Iron Condor Strategy Functions
###############################################################################
def fetch_option_contracts(symbol: str, expiration_date: str) -> List[Dict]:
    trader_for_options = AlpacaTrader(config, logger)
    return trader_for_options.get_options_contracts(symbol, expiration_date)

def select_iron_condor_legs(contracts: list, underlying_price: float, spread_width: float, trader: AlpacaTrader) -> Dict[str, Dict]:
    # Filter contracts for liquidity (volume >= 10)
    liquid_contracts = [c for c in contracts if c.get("volume", 0) >= 10]
    if not liquid_contracts:
        liquid_contracts = contracts

    # For each contract, obtain a valid live quote; only keep those with valid quotes.
    def with_quote(c):
        q = trader.get_latest_option_quote(c.get("symbol"))
        if q is not None:
            c["quote"] = q
            return True
        return False

    calls = [c for c in liquid_contracts if c.get("type", "").lower() == "call" and with_quote(c)]
    puts = [c for c in liquid_contracts if c.get("type", "").lower() == "put" and with_quote(c)]
    if not calls or not puts:
        raise ValueError("No liquid options with valid quotes found.")

    calls_above = sorted([c for c in calls if float(c["strike_price"]) >= underlying_price], key=lambda c: float(c["strike_price"]))
    puts_below = sorted([p for p in puts if float(p["strike_price"]) <= underlying_price], key=lambda p: float(p["strike_price"]), reverse=True)
    if not calls_above or not puts_below:
        raise ValueError("Could not find candidate contracts on both sides of the underlying price.")

    short_call = calls_above[0]
    short_put = puts_below[0]
    sc_strike = float(short_call["strike_price"])
    sp_strike = float(short_put["strike_price"])

    long_call_candidates = [c for c in calls_above if float(c["strike_price"]) >= sc_strike + spread_width]
    long_call = long_call_candidates[0] if long_call_candidates else calls_above[-1]

    long_put_candidates = [p for p in puts_below if float(p["strike_price"]) <= sp_strike - spread_width]
    long_put = long_put_candidates[0] if long_put_candidates else puts_below[-1]

    logger.info(f"Iron Condor Legs for underlying at {underlying_price}: Short Call @ {short_call['strike_price']} (quote: {short_call.get('quote')}), "
                f"Long Call @ {long_call['strike_price']} (quote: {long_call.get('quote')}), "
                f"Short Put @ {short_put['strike_price']} (quote: {short_put.get('quote')}), "
                f"Long Put @ {long_put['strike_price']} (quote: {long_put.get('quote')})")
    return {
        "short_call": short_call,
        "long_call": long_call,
        "short_put": short_put,
        "long_put": long_put
    }

async def execute_iron_condor(trader: AlpacaTrader, symbol: str, expiration_date: str, spread_width: float) -> List:
    symbol = symbol.upper()
    contracts = fetch_option_contracts(symbol, expiration_date)
    if not contracts:
        logger.error(f"No option contracts found for {symbol} expiring {expiration_date}.")
        return []
    try:
        underlying_price = trader.get_latest_price(symbol)
    except Exception as e:
        logger.error(f"Failed to retrieve underlying price for {symbol}: {e}")
        underlying_price = float(contracts[0]["strike_price"])
    if underlying_price is None:
        underlying_price = float(contracts[0]["strike_price"])
    logger.info(f"Underlying price for {symbol}: {underlying_price}")
    try:
        legs = select_iron_condor_legs(contracts, underlying_price, spread_width, trader)
    except ValueError as e:
        logger.error(f"Error selecting iron condor legs for {symbol}: {e}")
        return []
    orders = []
    order_specs = [
        {"contract": legs["short_call"], "side": "sell"},
        {"contract": legs["long_call"],  "side": "buy"},
        {"contract": legs["short_put"],  "side": "sell"},
        {"contract": legs["long_put"],   "side": "buy"}
    ]
    for spec in order_specs:
        contract = spec["contract"]
        side = spec["side"]
        contract_symbol = contract.get("symbol")
        try:
            logger.info(f"Placing {side.upper()} order for {contract_symbol} (strike {contract['strike_price']}, exp {contract['expiration_date']})")
            limit_price = await calculate_limit_price(trader, contract_symbol, side)
            if limit_price is None or limit_price <= 0:
                logger.error(f"Calculated limit price for {contract_symbol} is invalid: {limit_price}. Skipping order.")
                continue
            order = await trader.submit_order(
                symbol=contract_symbol,
                qty=1,
                side=side,
                type="limit",
                limit_price=limit_price
            )
            if order is not None:
                logger.info(f"Order placed for {contract_symbol}: {order.id}")
                orders.append(order)
            else:
                logger.error(f"Order for {contract_symbol} returned None.")
        except APIError as api_err:
            logger.error(f"Failed to place {side.upper()} order for {contract_symbol}: {api_err}")
    return orders

###############################################################################
# 5) Transformer Network Components for RL Agent
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()
    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.std_init * bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_sigma, self.std_init * bound)
    def reset_noise(self):
        eps_in = torch.randn(self.in_features)
        eps_out = torch.randn(self.out_features)
        eps_in = eps_in.sign().mul_(eps_in.abs().sqrt_())
        eps_out = eps_out.sign().mul_(eps_out.abs().sqrt_())
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class TransformerDQN(nn.Module):
    def __init__(self, in_dim, out_dim, num_clusters, num_layers=3, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(in_dim, 128)
        self.pos_encoder = PositionalEncoding(128, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=nhead, dim_feedforward=512, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cnn1 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.cnn3 = nn.Conv1d(128, 64, kernel_size=7, padding=3)
        self.attention = nn.MultiheadAttention(embed_dim=192, num_heads=4, dropout=0.1)
        self.cluster_layers = nn.ModuleList([
            nn.Sequential(
                NoisyLinear(192, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                NoisyLinear(128, 64)
            ) for _ in range(num_clusters)
        ])
        self.value_stream = NoisyLinear(64, 1)
        self.advantage_stream = NoisyLinear(64, out_dim)
        self.num_clusters = num_clusters
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    def forward(self, x, cluster_ids, history):
        batch_size, seq_len, _ = history.shape
        embedded = self.embedding(history)  # (B, seq_len, 128)
        embedded = embedded.transpose(0, 1)   # (seq_len, B, 128)
        embedded = self.pos_encoder(embedded)
        transformer_out = self.transformer(embedded)  # (seq_len, B, 128)
        cnn_input = transformer_out.permute(1, 2, 0)    # (B, 128, seq_len)
        cnn1_out = F.relu(self.cnn1(cnn_input)).max(dim=-1)[0]
        cnn2_out = F.relu(self.cnn2(cnn_input)).max(dim=-1)[0]
        cnn3_out = F.relu(self.cnn3(cnn_input)).max(dim=-1)[0]
        combined = torch.cat([cnn1_out, cnn2_out, cnn3_out], dim=1)  # (B, 192)
        combined_unsq = combined.unsqueeze(0)  # (1, B, 192)
        attn_out, _ = self.attention(combined_unsq, combined_unsq, combined_unsq)
        attn_out = attn_out.squeeze(0)  # (B, 192)
        cluster_outputs = []
        for i in range(batch_size):
            cid = 0
            cluster_outputs.append(self.cluster_layers[cid](attn_out[i].unsqueeze(0)))
        cluster_out = torch.cat(cluster_outputs, dim=0)  # (B, 64)
        values = self.value_stream(cluster_out)
        advantages = self.advantage_stream(cluster_out)
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

###############################################################################
# 5) Prioritized Replay Memory for RL Training
###############################################################################
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'cluster_id', 'history', 'next_history'))

class PrioritizedReplayMemory:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    def push(self, transition: tuple):
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size: int, beta: float = 0.4):
        if not self.memory:
            return [], [], []
        prios = self.priorities[:self.pos] if len(self.memory) < self.capacity else self.priorities
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights.tolist()
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
    def __len__(self):
        return len(self.memory)

class PrioritizedReplayMemoryPersistence:
    def __init__(self, filepath: str, logger: logging.Logger):
        self.filepath = filepath
        self.logger = logger
    def save_memory(self, memory: PrioritizedReplayMemory):
        try:
            with open(self.filepath, 'wb') as f:
                pickle.dump({'memory': memory.memory, 'priorities': memory.priorities, 'pos': memory.pos}, f)
            self.logger.info(f"Saved replay memory to {self.filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save replay memory: {e}", exc_info=True)
    def load_memory(self, memory: PrioritizedReplayMemory):
        if not os.path.exists(self.filepath):
            self.logger.warning(f"No replay memory file found at {self.filepath}. Starting fresh.")
            return
        try:
            with open(self.filepath, 'rb') as f:
                data = pickle.load(f)
                memory.memory = data['memory']
                memory.priorities = data['priorities']
                memory.pos = data['pos']
            self.logger.info(f"Loaded replay memory from {self.filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load replay memory: {e}", exc_info=True)

###############################################################################
# 6) TransformerAgent (RL Agent for Options Signals)
###############################################################################
class TransformerAgent:
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_clusters: int,
        device: torch.device,
        memory_size: int,
        batch_size: int,
        gamma: float,
        learning_rate: float,
        epsilon_decay: int,
        alpha: float,
        beta_start: float,
        logger: logging.Logger,
        num_layers: int = 3,
        nhead: int = 4,
        confidence_threshold: float = 0.1
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_clusters = num_clusters
        self.device = device
        self.logger = logger
        self.confidence_threshold = confidence_threshold
        self.policy_net = TransformerDQN(in_dim, out_dim, num_clusters, num_layers, nhead).to(device)
        self.target_net = TransformerDQN(in_dim, out_dim, num_clusters, num_layers, nhead).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = PrioritizedReplayMemory(memory_size, alpha=alpha)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.steps_done = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.beta_start = beta_start
        self.beta_frames = 100000
    def select_action(self, state: torch.Tensor, history: torch.Tensor, cluster_id: int) -> int:
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        self.policy_net.reset_noise()
        if random.random() < epsilon:
            self.logger.debug(f"Random action taken due to epsilon {epsilon:.4f}")
            return random.randrange(self.out_dim)
        else:
            with torch.no_grad():
                cids = torch.tensor([cluster_id], dtype=torch.long, device=self.device)
                q_vals = self.policy_net(state, cids, history)
                top2 = q_vals.topk(2, dim=1)
                margin = top2.values[0, 0] - top2.values[0, 1]
                if margin < self.confidence_threshold:
                    self.logger.info("Low confidence (margin {:.4f}). Forcing Hold action.".format(margin))
                    return 0
                action = q_vals.argmax(dim=1).item()
                self.logger.debug(f"Selected action {action} with Q-values {q_vals}")
                return action
    def store_transition(self, s, a, r, ns, d, cid, hist, n_hist):
        self.memory.push((s, a, r, ns, d, cid, hist, n_hist))
    def optimize_model(self, beta: float):
        if len(self.memory) < self.batch_size:
            return
        transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        if not transitions:
            return
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)
        cluster_ids = torch.tensor(batch.cluster_id, dtype=torch.long).to(self.device)
        history_batch = torch.cat(batch.history).to(self.device)
        next_history_batch = torch.cat(batch.next_history).to(self.device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.policy_net.reset_noise()
        current_q_values = self.policy_net(state_batch, cluster_ids, history_batch).gather(1, action_batch)
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch, cluster_ids, next_history_batch).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_state_batch, cluster_ids, next_history_batch).gather(1, next_actions)
            target_q = reward_batch + (1.0 - done_batch) * self.gamma * next_q_values
        td_errors = current_q_values - target_q
        loss = (td_errors.pow(2) * weights_tensor).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-5
        self.memory.update_priorities(indices, new_priorities.flatten())
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.logger.info("Updated target network.")
    def save_checkpoint(self, filepath: str):
        try:
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'steps_done': self.steps_done
            }, filepath)
            self.logger.info(f"Saved checkpoint => {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
    def load_checkpoint(self, filepath: str):
        if not os.path.exists(filepath):
            self.logger.warning(f"No checkpoint file {filepath}. Starting fresh.")
            return
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'], strict=False)
            self.target_net.load_state_dict(checkpoint['target_net'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps_done = checkpoint.get('steps_done', 0)
            self.logger.info(f"Loaded checkpoint => {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}", exc_info=True)

###############################################################################
# 7) Build State and History from Bars
###############################################################################
def build_state(df: pd.DataFrame) -> torch.Tensor:
    if df.empty:
        return torch.zeros((1, 19), dtype=torch.float32).to(device)
    last = df.iloc[-1]
    feats = [last.get(col, 0.0) for col in [
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_bbm', 'bb_bbh', 'bb_bbl', 'atr', 'adx',
        'keltner_upper', 'keltner_lower', 'fib_23.6', 'fib_38.2',
        'vw_macd', 'momentum_3d', 'hour_sin', 'hour_cos',
        'day_sin', 'day_cos'
    ]]
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)

def build_history(df: pd.DataFrame, window: int = 10) -> torch.Tensor:
    if df.empty:
        return torch.zeros((1, window, 19), dtype=torch.float32).to(device)
    if len(df) < window:
        df_window = pd.concat([df.iloc[-1:]] * window)
    else:
        df_window = df.iloc[-window:]
    seq = []
    for _, row in df_window.iterrows():
        seq.append([row.get(col, 0.0) for col in [
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_bbm', 'bb_bbh', 'bb_bbl', 'atr', 'adx',
            'keltner_upper', 'keltner_lower', 'fib_23.6', 'fib_38.2',
            'vw_macd', 'momentum_3d', 'hour_sin', 'hour_cos',
            'day_sin', 'day_cos'
        ]])
    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

###############################################################################
# 8) Options Module: Dynamic Options Screening, Iron Condor Execution & Order Placement
###############################################################################
def next_friday_expiration() -> datetime:
    today = datetime.now(timezone.utc)
    days_ahead = 4 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    next_friday = today + timedelta(days=days_ahead)
    return next_friday.replace(hour=16, minute=0, second=0, microsecond=0)

def build_option_symbol(underlying: str, underlying_price: float, is_call: bool) -> str:
    expiration = next_friday_expiration()
    strike = int(round(underlying_price, 2) * 1000)
    option_type = 'C' if is_call else 'P'
    return f"{underlying}{expiration.strftime('%y%m%d')}{option_type}{strike:08d}"

async def process_options(underlying: str, trader: AlpacaTrader, logger: logging.Logger, side: str):
    underlying_price = trader.get_latest_price(underlying)
    logger.debug(f"Latest price for {underlying}: {underlying_price}")
    if underlying_price is None:
        logger.error(f"Could not retrieve latest price for {underlying}.")
        return
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    options_chain = trader.get_options_contracts(underlying, tomorrow)
    logger.debug(f"Options chain for {underlying}: {options_chain}")
    if not options_chain:
        logger.error(f"No options contracts found for {underlying}.")
        return
    desired_letter = "C" if side=="long" else "P"
    filtered = [c for c in options_chain if (c.get("option_type", c.get("contract_type", "")).upper().startswith(desired_letter))]
    if not filtered:
        if options_chain:
            logger.debug(f"Available keys in first options contract for {underlying}: {list(options_chain[0].keys())}")
        logger.error(f"No {side} options found for {underlying}.")
        return
    def parse_strike(contract):
        if "strike_price" in contract:
            return float(contract["strike_price"])
        else:
            try:
                strike_str = contract.get("symbol")[-8:]
                return int(strike_str) / 100.0
            except Exception as e:
                logger.error(f"Error parsing strike for contract {contract.get('symbol')}: {e}")
                return float('inf')
    try:
        best_contract = min(filtered, key=lambda c: abs(parse_strike(c) - underlying_price))
    except Exception as e:
        logger.error(f"Error selecting best contract: {e}")
        return
    chosen_strike = parse_strike(best_contract)
    generated_symbol = best_contract.get("symbol")
    logger.info(f"Selected option for {underlying} ({side}): {generated_symbol} with strike {chosen_strike}")
    limit_price = await calculate_limit_price(trader, generated_symbol, "buy")
    if limit_price is None or limit_price <= 0:
        logger.error(f"Calculated limit price for {generated_symbol} is invalid: {limit_price}. Skipping order.")
        return
    order = await trader.submit_order(
        symbol=generated_symbol,
        qty=1,
        side="buy",
        type="limit",
        limit_price=limit_price
    )
    if order:
        logger.info(f"Executed options order for {generated_symbol}")
        rl_instance.save_agent()

async def calculate_limit_price(trader: AlpacaTrader, symbol: str, side: str, tolerance: float = 0.01) -> Optional[float]:
    price = trader.get_latest_price(symbol)
    if price is None or price <= 0:
        logger.error(f"Invalid latest price for {symbol}: {price}")
        return None
    if side.lower() == "buy":
        return round(price * (1 + tolerance), 2)
    else:
        return round(price * (1 - tolerance), 2)

async def process_stock_trade(underlying: str, trader: AlpacaTrader, logger: logging.Logger, side: str):
    latest_price = trader.get_latest_price(underlying)
    if latest_price is None:
        logger.error(f"Could not retrieve latest price for {underlying}.")
        return
    limit_price = await calculate_limit_price(trader, underlying, side)
    if limit_price is None or limit_price <= 0:
        logger.error(f"Calculated limit price for {underlying} is invalid: {limit_price}. Skipping order.")
        return
    order = await trader.submit_order(
         symbol=underlying,
         qty=1,
         side=side,
         type="limit",
         limit_price=limit_price
    )
    if order:
         logger.info(f"Executed stock order for {underlying} at limit price {limit_price}")
    else:
         logger.error(f"Failed to execute stock order for {underlying}")

###############################################################################
# 9) Options Trading Engine: Process Underlying & Execute Option Trade / Iron Condor
###############################################################################
def compute_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True, drop=False)
    df.sort_index(inplace=True)
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()
    atr_ind = AverageTrueRange(df['high'], df['low'], df['close'], window=ATR_PERIOD)
    df['atr'] = atr_ind.average_true_range()
    adx = ADXIndicator(df['high'], df['low'], df['close'], 14)
    df['adx'] = adx.adx()
    kelt = KeltnerChannel(df['high'], df['low'], df['close'], window=20)
    df['keltner_upper'] = kelt.keltner_channel_hband()
    df['keltner_lower'] = kelt.keltner_channel_lband()
    rolling_high = df['high'].rolling(50).max()
    rolling_low = df['low'].rolling(50).min()
    df['fib_23.6'] = rolling_high - (rolling_high - rolling_low) * 0.236
    df['fib_38.2'] = rolling_high - (rolling_high - rolling_low) * 0.382
    vwma_26 = (df['volume'] * df['close']).rolling(26).sum() / df['volume'].rolling(26).sum()
    vwma_9 = vwma_26.ewm(span=9, adjust=False).mean()
    df['vw_macd'] = vwma_26 - vwma_9
    df['momentum_3d'] = df['close'].pct_change(3)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.0)
    df = df.ffill().bfill().dropna()
    for col in ['rsi','macd','macd_signal','macd_diff','bb_bbm','bb_bbh','bb_bbl','atr','adx',
                'keltner_upper','keltner_lower','fib_23.6','fib_38.2','vw_macd','momentum_3d',
                'hour_sin','hour_cos','day_sin','day_cos']:
        if col in df.columns:
            lower = df[col].quantile(0.05)
            upper = df[col].quantile(0.95)
            df[col] = df[col].clip(lower, upper)
            cmin = df[col].min()
            cmax = df[col].max() + 1e-6
            df[col] = (df[col] - cmin) / (cmax - cmin)
    return df

async def process_symbol(symbol: str, rl_trader, trader: AlpacaTrader, logger: logging.Logger):
    df = compute_enhanced_indicators(pd.DataFrame(live_bars[symbol]))
    if df.empty:
        logger.info(f"{symbol} => Insufficient data for trading.")
        return
    state = build_state(df).to(rl_trader.device)
    hist_tensor = build_history(df, window=10).to(rl_trader.device)
    cluster_id = 0
    rl_action = rl_trader.pick_action(state, hist_tensor, cluster_id)
    if rl_action == 0:
        logger.info(f"{symbol} => RL Trader: Hold. No trade executed.")
        return
    elif rl_action in [1, 2]:
        if TRADE_OPTIONS:
            if rl_action == 1:
                logger.info(f"{symbol} => RL Trader Signal: Buy Call (long) option.")
                await process_options(symbol, trader, logger, "long")
            elif rl_action == 2:
                logger.info(f"{symbol} => RL Trader Signal: Buy Put (short) option.")
                await process_options(symbol, trader, logger, "short")
        else:
            if rl_action == 1:
                logger.info(f"{symbol} => RL Trader Signal: Buy underlying stock (bullish).")
                await process_stock_trade(symbol, trader, logger, "buy")
            elif rl_action == 2:
                logger.info(f"{symbol} => RL Trader Signal: Sell underlying stock (bearish).")
                await process_stock_trade(symbol, trader, logger, "sell")
    elif rl_action == 3:
        if TRADE_OPTIONS:
            spread_width = 10.0
            logger.info(f"{symbol} => RL Trader Signal: Execute Iron Condor strategy with spread width {spread_width}.")
            orders = await execute_iron_condor(trader, symbol, next_friday_expiration().strftime("%Y-%m-%d"), spread_width)
            if orders:
                logger.info(f"Iron Condor orders executed for {symbol}.")
        else:
            logger.info(f"{symbol} => Iron Condor signal received but options trading is disabled in config.")
    current_price = trader.get_latest_price(symbol)
    reward = get_reward(symbol, current_price, trade_state_instance) if current_price else 0.0
    next_state = build_state(df).to(rl_trader.device)
    next_history = build_history(df, window=10).to(rl_trader.device)
    rl_trader.store_transition(state, rl_action, reward, next_state, 0, cluster_id, hist_tensor, next_history)
    rl_trader.update_risk_mgr(trade_success=(reward > 0))

###############################################################################
# 10) Trade State and Reward
###############################################################################
class TradeState:
    def __init__(self):
        self.active_trades = {}
    def update_trade(self, symbol: str, entry_price: float, side: str):
        now = datetime.now(timezone.utc)
        if symbol not in self.active_trades:
            if side == 'long':
                stop_loss = entry_price * (1 - TRAILING_STOP_LOSS)
                take_profit = entry_price * (1 + TAKE_PROFIT)
            else:
                stop_loss = entry_price * (1 + TRAILING_STOP_LOSS)
                take_profit = entry_price * (1 - TAKE_PROFIT)
            self.active_trades[symbol] = {
                'entry_price': entry_price,
                'side': side,
                'entry_time': now,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
    def adjust_stop_loss(self, symbol: str, current_price: float):
        if symbol not in self.active_trades:
            return
        trade = self.active_trades[symbol]
        side = trade['side']
        if side == 'long' and current_price > trade['entry_price']:
            new_sl = current_price * (1 - TRAILING_STOP_LOSS)
            if new_sl > trade['stop_loss']:
                trade['stop_loss'] = new_sl
        elif side == 'short' and current_price < trade['entry_price']:
            new_sl = current_price * (1 + TRAILING_STOP_LOSS)
            if new_sl < trade['stop_loss']:
                trade['stop_loss'] = new_sl
    def check_exit(self, symbol: str, current_price: float, current_time: datetime, min_hold_days: int, max_hold_days: int) -> Optional[str]:
        if symbol not in self.active_trades:
            return None
        trade = self.active_trades[symbol]
        hold_duration = (current_time - trade['entry_time']).days
        side = trade['side']
        if side == 'long':
            if current_price <= trade['stop_loss']:
                del self.active_trades[symbol]
                return 'stop_loss'
            elif current_price >= trade['take_profit']:
                del self.active_trades[symbol]
                return 'take_profit'
        else:
            if current_price >= trade['stop_loss']:
                del self.active_trades[symbol]
                return 'stop_loss'
            elif current_price <= trade['take_profit']:
                del self.active_trades[symbol]
                return 'take_profit'
        if hold_duration >= max_hold_days:
            del self.active_trades[symbol]
            return 'time_exit'
        if hold_duration < min_hold_days:
            return None
        return None

def get_reward(symbol: str, current_price: float, trade_state: TradeState) -> float:
    if symbol in trade_state.active_trades:
        trade = trade_state.active_trades[symbol]
        entry_price = trade['entry_price']
        side = trade['side']
        return (current_price - entry_price) / entry_price if side == 'long' else (entry_price - current_price) / entry_price
    return 0.0

###############################################################################
# 11) Live Bars Storage and WebSocket Callback
###############################################################################
async def on_bar_callback(bar):
    if not is_market_hours():
        logger.info("Market is closed. Skipping bar processing.")
        return
    timestamp = getattr(bar, "timestamp", bar._raw.get("timestamp"))
    open_price = getattr(bar, "open", bar._raw.get("open"))
    high_price = getattr(bar, "high", bar._raw.get("high"))
    low_price = getattr(bar, "low", bar._raw.get("low"))
    close_price = getattr(bar, "close", bar._raw.get("close"))
    volume = getattr(bar, "volume", bar._raw.get("volume"))
    symbol = bar.symbol
    bar_dict = {
        "timestamp": timestamp,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume
    }
    if symbol not in live_bars:
        live_bars[symbol] = []
    live_bars[symbol].append(bar_dict)
    if len(live_bars[symbol]) > 50:
        live_bars[symbol] = live_bars[symbol][-50:]
    if len(live_bars[symbol]) >= 10:
        asyncio.create_task(process_symbol(symbol, rl_instance, trader_instance, logger))

###############################################################################
# 12) Options WebSocket Callback for Real-Time Option Data
###############################################################################
def on_option_message(msg):
    if msg.get("T") == "t":
        logger.info(f"Option Trade: {msg}")
    elif msg.get("T") == "q":
        logger.info(f"Option Quote: {msg}")
    elif msg.get("T") == "error":
        logger.error(f"Option Stream Error: {msg}")

###############################################################################
# 13) RL Optimization Loop (Optional Training)
###############################################################################
async def rl_optimization_loop():
    episode = 0
    while True:
        try:
            beta = min(1.0, rl_instance.agent.beta_start + episode * (1.0 - rl_instance.agent.beta_start) / rl_instance.agent.beta_frames)
            rl_instance.optimize(beta)
            if episode % TARGET_UPDATE == 0:
                rl_instance.update_target()
            if episode % 100 == 0:
                rl_instance.save_agent()
            episode += 1
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error in RL optimization loop: {e}", exc_info=True)
            await asyncio.sleep(60)

###############################################################################
# 14) RLTrader Wrapper for Options Trading and Risk Management
###############################################################################
class RLTrader:
    def __init__(self, device: torch.device, logger: logging.Logger,
                 checkpoint_path: str = "agent_checkpoint.pth", replay_memory_path: str = "replay_memory.pkl"):
        self.device = device
        self.logger = logger
        self.state_dim = 19
        self.action_dim = 4  # 0=Hold, 1=Buy Call, 2=Buy Put, 3=Execute Iron Condor
        self.agent = TransformerAgent(
            in_dim=self.state_dim,
            out_dim=self.action_dim,
            num_clusters=10,
            device=device,
            memory_size=MEMORY_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_rate=LEARNING_RATE,
            epsilon_decay=EPSILON_DECAY,
            alpha=ALPHA,
            beta_start=BETA_START,
            logger=logger,
            num_layers=TRANSFORMER_LAYERS,
            nhead=ATTENTION_HEADS,
            confidence_threshold=0.1
        )
        self.checkpoint_path = checkpoint_path
        self.replay_memory_path = replay_memory_path
        self.memory_persistence = PrioritizedReplayMemoryPersistence(self.replay_memory_path, logger)
        self.load_agent()
        self.risk_manager = EnhancedRiskManager()
    def load_agent(self):
        self.agent.load_checkpoint(self.checkpoint_path)
        self.memory_persistence.load_memory(self.agent.memory)
    def save_agent(self):
        self.agent.save_checkpoint(self.checkpoint_path)
        self.memory_persistence.save_memory(self.agent.memory)
    def pick_action(self, state: torch.Tensor, history: torch.Tensor, cluster_id: int) -> int:
        return self.agent.select_action(state, history, cluster_id)
    def store_transition(self, *args):
        self.agent.store_transition(*args)
    def optimize(self, beta: float):
        self.agent.optimize_model(beta)
    def update_target(self):
        self.agent.update_target_net()
    def update_risk_mgr(self, trade_success: bool):
        self.risk_manager.update_risk_parameters(trade_success)
    def get_position_size(self, capital: float, entry_price: float, atr: float, correlation_matrix=None) -> int:
        return self.risk_manager.get_position_size(capital, entry_price, atr, correlation_matrix)

###############################################################################
# 15) Enhanced Risk Manager Definitions
###############################################################################
class AdaptiveRiskManager:
    def __init__(self):
        self.win_rate = 0.5
        self.consecutive_losses = 0
        self.risk_multiplier = 1.0
    def update_risk_parameters(self, trade_success: bool):
        if trade_success:
            self.consecutive_losses = 0
            self.win_rate = 0.9 * self.win_rate + 0.1
        else:
            self.consecutive_losses += 1
            self.win_rate = 0.9 * self.win_rate
        if self.consecutive_losses > 2:
            self.risk_multiplier = max(0.5, 1.0 - 0.1 * self.consecutive_losses)
        else:
            self.risk_multiplier = min(2.0, 1.0 + (self.win_rate - 0.5))

class EnhancedRiskManager(AdaptiveRiskManager):
    def get_position_size(self, capital, entry_price, atr, correlation_matrix):
        if correlation_matrix is None or len(correlation_matrix) == 0:
            corr_penalty = 1.0
        else:
            max_corr = np.max(np.abs(correlation_matrix))
            corr_penalty = 1 - max_corr
        risk_multiplier = self.risk_multiplier * corr_penalty
        dynamic_atr_mult = VOL_TRAILING_MULTIPLIER if atr > entry_price * 0.01 else 1.8
        risk_amount = capital * RISK_PER_TRADE * risk_multiplier
        dollar_risk = atr * entry_price * dynamic_atr_mult
        if dollar_risk <= 0:
            return 0
        shares = int(risk_amount / dollar_risk)
        return max(shares, 0)

###############################################################################
# 16) Main Function: Initialization, WebSocket Subscriptions & Option Stream
###############################################################################
global_symbols: List[str] = []

trader_instance: Optional[AlpacaTrader] = None
rl_instance: Optional[RLTrader] = None
trade_state_instance: Optional[TradeState] = None

async def main():
    global trader_instance, rl_instance, trade_state_instance, global_symbols
    trader_instance = AlpacaTrader(config, logger)
    if not is_market_hours():
        seconds_to_wait = time_until_market_open()
        logger.info(f"Market is closed. Waiting for {seconds_to_wait:.0f} seconds until market open.")
        await asyncio.sleep(seconds_to_wait)
    # Retrieve all tradable symbols.
    all_symbols = get_all_tradable_symbols(trader_instance, logger)
    candidate_symbols = []
    metrics = {}
    for sym in all_symbols:
        df_hist = await fetch_initial_bars(sym, trader_instance, logger, days=90, timeframe="15Min")
        if df_hist.empty or len(df_hist) < 10:
            logger.warning(f"{sym} => Skipped due to insufficient historical bars.")
            continue
        live_bars[sym] = df_hist.to_dict(orient="records")
        df_hist['return'] = df_hist['close'].pct_change()
        win_ratio = (df_hist['return'] > 0).mean()
        avg_vol = df_hist['volume'].mean()
        metrics[sym] = (win_ratio, avg_vol)
        candidate_symbols.append(sym)
    volumes = [metrics[sym][1] for sym in candidate_symbols]
    min_vol, max_vol = min(volumes), max(volumes)
    combined_scores = {}
    for sym in candidate_symbols:
        win_ratio, avg_vol = metrics[sym]
        norm_vol = (avg_vol - min_vol) / (max_vol - min_vol) if max_vol > min_vol else 0
        combined_scores[sym] = win_ratio + norm_vol
    top_50 = sorted(candidate_symbols, key=lambda x: combined_scores[x], reverse=True)[:50]
    global_symbols = top_50
    logger.info(f"Selected top {len(global_symbols)} symbols for trading based on win ratio and moving volume.")
    rl_instance = RLTrader(device, logger)
    trade_state_instance = TradeState()
    stream = Stream(API_KEY, API_SECRET, base_url=BASE_URL, data_feed="iex")
    stream.subscribe_bars(on_bar_callback, *global_symbols)
    asyncio.create_task(rl_optimization_loop())
    stream.run()

###############################################################################
# Program Entry Point
###############################################################################
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Saving agent and exiting.")
        if rl_instance:
            rl_instance.save_agent()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        if rl_instance:
            rl_instance.save_agent()
