create table if not exists board_intraday_events (
    trade_date varchar not null,
    code varchar not null,
    first_limit_time varchar not null,
    last_limit_time varchar,
    seal_duration_seconds double,
    reopen_count integer,
    limit_up double not null,
    close double,
    is_close_sealed boolean not null,
    source varchar,
    ingested_at varchar,
    primary key (trade_date, code)
);

create table if not exists board_order_book_snapshots (
    trade_date varchar not null,
    code varchar not null,
    snapshot_time varchar not null,
    bid_price_1 double,
    bid_volume_1 double,
    ask_price_1 double,
    ask_volume_1 double,
    limit_queue_volume double,
    source varchar,
    ingested_at varchar,
    primary key (trade_date, code, snapshot_time)
);

create table if not exists board_order_fills (
    trade_date varchar not null,
    code varchar not null,
    signal_time varchar not null,
    order_time varchar not null,
    side varchar not null,
    order_price double,
    order_qty double,
    filled_qty double,
    avg_fill_price double,
    status varchar,
    source varchar,
    ingested_at varchar,
    primary key (trade_date, code, signal_time, order_time, side)
);
