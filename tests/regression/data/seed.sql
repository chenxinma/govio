-- observe load 回归种子数据（DuckDB）
-- customers：5 行，覆盖 NULL / 中文 / 空串 / 单引号 / 负数 边界
-- orders：5 行，覆盖 NULL 金额 / 孤儿外键 / 日期
-- 列名与行值与 tests/regression/test_load_cli.py 的 L01-L05 断言一一对应

CREATE TABLE customers (
    customer_id INTEGER,   -- 客户编号
    name VARCHAR,          -- 客户姓名（含中文、空串、单引号边界）
    region VARCHAR,        -- 地区（含 NULL）
    credit DOUBLE          -- 信用额度（含负数、NULL）
);

INSERT INTO customers (customer_id, name, region, credit) VALUES
    (1, '张三', '华东', 100.0),
    (2, '李四', NULL, 50.0),            -- records[1].region 期望 NULL
    (3, '王五', '华南', -20.25),        -- records[2].credit 期望 -20.25（负数）
    (4, 'O''Brien 李', '华北', NULL),   -- records[3] 期望引号姓名 + credit NULL
    (5, '', '西南', 0.0);               -- records[4].name 期望空串

CREATE TABLE orders (
    order_id INTEGER,      -- 订单编号
    customer_id INTEGER,   -- 客户编号（1005 为孤儿外键，无对应客户）
    amount DOUBLE,         -- 订单金额（1002 为 NULL）
    status VARCHAR,        -- 订单状态
    created_at DATE        -- 创建日期
);

INSERT INTO orders (order_id, customer_id, amount, status, created_at) VALUES
    (1001, 1, 120.5, 'paid', DATE '2024-01-15'),
    (1002, 1, NULL, 'pending', DATE '2024-02-20'),
    (1003, 2, 80.0, 'paid', DATE '2024-03-10'),
    (1004, 3, 200.0, 'cancelled', DATE '2024-04-05'),
    (1005, 9999, 55.5, 'paid', DATE '2024-05-01');
