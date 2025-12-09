# ==========================
# Deephaven + Kafka + Protobuf trades
# ==========================
from __future__ import annotations

import threading
from typing import Iterator, NamedTuple

from confluent_kafka import Consumer, KafkaException, KafkaError

# Deephaven imports
from deephaven.stream.table_publisher import table_publisher
from deephaven.execution_context import get_exec_ctx
from deephaven import dtypes as dht
from deephaven import new_table
from deephaven.column import long_col, string_col, double_col

# Your generated protobuf classes
from trades_pb2 import Trades  # message Trades { repeated Trade trades = 1; }


# --------------------------
# 1. Protobuf decoder (iterator-based)
# --------------------------

class TradeRecord(NamedTuple):
    ts: int        # timestamp (e.g. epoch ns/ms) - adjust as needed
    sym: str       # symbol
    qty: int       # signed quantity
    price: float   # price
    mmt: bytes | str  # mmt flag string/binary


# Reuse a single Trades() instance for efficiency (single-threaded decoder)
_trades_msg = Trades()


def decode_trades(raw: bytes) -> Iterator[TradeRecord]:
    """
    Decode a single Kafka message payload (bytes) containing a Trades protobuf,
    and yield one TradeRecord per inner Trade.
    """
    _trades_msg.Clear()
    _trades_msg.ParseFromString(raw)

    for t in _trades_msg.trades:
        yield TradeRecord(
            ts=t.ts,
            sym=t.sym,
            qty=t.qty,
            price=t.price,
            mmt=t.mmt,
        )


# --------------------------
# 2. Deephaven table publisher setup
# --------------------------

# Define the schema of the live trades table
coldefs = {
    "Ts": dht.long,      # we'll store ts as a long; you can convert to DateTime later
    "Sym": dht.string,
    "Qty": dht.long,
    "Price": dht.double,
    "MMT": dht.string,
}

# Create the blink table + publisher
trades_blink, trades_publisher = table_publisher(
    name="Kafka Trades",
    col_defs=coldefs,
)

# Optional: append-only history table (keeps all rows)
from deephaven.stream import blink_to_append_only

trades_history = blink_to_append_only(trades_blink)


def trades_to_table(records: list[TradeRecord]):
    """
    Convert a list of TradeRecord into a Deephaven table matching `coldefs`.
    This is used to batch inserts into the table publisher.
    """
    if not records:
        from deephaven import empty_table
        return empty_table(0)

    ts_vals = [r.ts for r in records]
    sym_vals = [r.sym for r in records]
    qty_vals = [r.qty for r in records]
    price_vals = [r.price for r in records]

    # Normalize MMT to string for display
    mmt_vals = [
        r.mmt.decode("ascii", errors="ignore") if isinstance(r.mmt, (bytes, bytearray)) else str(r.mmt)
        for r in records
    ]

    return new_table(
        [
            long_col("Ts", ts_vals),
            string_col("Sym", sym_vals),
            long_col("Qty", qty_vals),
            double_col("Price", price_vals),
            string_col("MMT", mmt_vals),
        ]
    )


# --------------------------
# 3. Kafka consumer loop feeding Deephaven
# --------------------------

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"   # <-- change for your cluster
KAFKA_TOPIC = "trades-topic"                 # <-- change to your topic name


def kafka_consumer_loop():
    """
    Run in a background thread inside Deephaven.
    Polls Kafka, decodes protobuf Trades messages, and publishes to Deephaven.
    """
    conf = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": "deephaven-trades-consumer",
        "auto.offset.reset": "earliest",  # or "latest" if you only want new trades
    }

    consumer = Consumer(conf)
    consumer.subscribe([KAFKA_TOPIC])

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition, not fatal; just continue
                    continue
                raise KafkaException(msg.error())

            raw = msg.value()
            if raw is None:
                continue

            # Decode protobuf -> TradeRecord iterator -> materialize into a list batch
            batch: list[TradeRecord] = list(decode_trades(raw))
            if not batch:
                continue

            # Convert to a Deephaven table and publish
            batch_table = trades_to_table(batch)
            trades_publisher.add(batch_table)

    finally:
        consumer.close()


# --------------------------
# 4. Start the consumer in a Deephaven-safe thread
# --------------------------

# Deephaven requires table operations from threads to run in an ExecutionContext
ctx = get_exec_ctx()


def thread_func():
    with ctx:
        kafka_consumer_loop()


# Start the background thread
kafka_thread = threading.Thread(target=thread_func, name="KafkaTradesConsumer", daemon=True)
kafka_thread.start()

# At this point:
# - `trades_blink` shows the latest trades (blink table)
# - `trades_history` accumulates all trades (append-only)
# You can inspect and operate on these tables in Deephaven normally.
