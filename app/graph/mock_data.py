from __future__ import annotations

from datetime import date, timedelta

MOCK_TENANTS: dict[str, dict] = {
    "megamart": {"active": True, "support_email": "support@megamart.com"},
    "tiny-candles": {"active": True, "support_email": "hello@tinycandles.in"},
    "inactive-co": {"active": False, "support_email": "support@inactive.com"},
}

MOCK_ORDERS: dict[str, dict[str, dict]] = {
    "megamart": {
        "ORD-MM-1001": {
            "order_id": "ORD-MM-1001",
            "customer_email": "alice@example.com",
            "category": "electronics",
            "product": "Wireless Headphones",
            "amount": 4500,
            "total_amount": 4500,
            "status": "delivered",
            "order_status": "delivered",
            "order_date": "2026-09-10",
            "delivery_date": "2026-09-20",
            "tracking_number": "TRK-MM-100101",
            "shipping_carrier": "BlueDart",
            "items": [
                {
                    "item_id": "ITM-1001-1",
                    "product": "Wireless Headphones",
                    "category": "electronics",
                    "amount": 4500,
                    "status": "delivered",
                },
            ],
        },
        "ORD-MM-1002": {
            "order_id": "ORD-MM-1002",
            "customer_email": "bob@example.com",
            "category": "clothing",
            "product": "Cotton T-Shirt",
            "amount": 1200,
            "total_amount": 1200,
            "status": "delivered",
            "order_status": "delivered",
            "order_date": "2026-09-05",
            "delivery_date": "2026-09-15",
            "tracking_number": "TRK-MM-100201",
            "shipping_carrier": "Delhivery",
            "items": [
                {
                    "item_id": "ITM-1002-1",
                    "product": "Cotton T-Shirt",
                    "category": "clothing",
                    "amount": 1200,
                    "status": "delivered",
                },
            ],
        },
        "ORD-MM-1003": {
            "order_id": "ORD-MM-1003",
            "customer_email": "charlie@example.com",
            "category": "perishables",
            "product": "Organic Fruit Box",
            "amount": 850,
            "total_amount": 850,
            "status": "delivered",
            "order_status": "delivered",
            "order_date": "2026-09-25",
            "delivery_date": "2026-09-28",
            "tracking_number": "TRK-MM-100301",
            "shipping_carrier": "BlueDart",
            "items": [
                {
                    "item_id": "ITM-1003-1",
                    "product": "Organic Fruit Box",
                    "category": "perishables",
                    "amount": 850,
                    "status": "delivered",
                },
            ],
        },
        "ORD-MM-1004": {
            "order_id": "ORD-MM-1004",
            "customer_email": "dave@example.com",
            "category": "electronics",
            "product": "4K OLED TV",
            "amount": 65000,
            "total_amount": 65000,
            "status": "delivered",
            "order_status": "delivered",
            "order_date": "2026-09-05",
            "delivery_date": "2026-09-10",
            "tracking_number": "TRK-MM-100401",
            "shipping_carrier": "DTDC",
            "items": [
                {
                    "item_id": "ITM-1004-1",
                    "product": "4K OLED TV",
                    "category": "electronics",
                    "amount": 65000,
                    "status": "delivered",
                },
            ],
        },
        "ORD-MM-1005": {
            "order_id": "ORD-MM-1005",
            "customer_email": "alice@example.com",
            "category": "clothing",
            "product": "Winter Jacket",
            "amount": 3800,
            "total_amount": 3800,
            "status": "processing",
            "order_status": "processing",
            "order_date": "2026-10-02",
            "delivery_date": None,
            "tracking_number": None,
            "shipping_carrier": None,
            "items": [
                {
                    "item_id": "ITM-1005-1",
                    "product": "Winter Jacket",
                    "category": "clothing",
                    "amount": 3800,
                    "status": "processing",
                },
            ],
        },
        "ORD-MM-1006": {
            "order_id": "ORD-MM-1006",
            "customer_email": "eve@example.com",
            "category": "electronics",
            "product": "Bluetooth Speaker",
            "amount": 2500,
            "total_amount": 2500,
            "status": "shipped",
            "order_status": "shipped",
            "order_date": "2026-09-30",
            "delivery_date": None,
            "tracking_number": "TRK-MM-100601",
            "shipping_carrier": "BlueDart",
            "items": [
                {
                    "item_id": "ITM-1006-1",
                    "product": "Bluetooth Speaker",
                    "category": "electronics",
                    "amount": 2500,
                    "status": "shipped",
                },
            ],
        },
        "ORD-MM-1007": {
            "order_id": "ORD-MM-1007",
            "customer_email": "frank@example.com",
            "category": "electronics",
            "product": "USB-C Hub",
            "amount": 1800,
            "total_amount": 1800,
            "status": "pending",
            "order_status": "pending",
            "order_date": "2026-10-03",
            "delivery_date": None,
            "tracking_number": None,
            "shipping_carrier": None,
            "items": [
                {
                    "item_id": "ITM-1007-1",
                    "product": "USB-C Hub",
                    "category": "electronics",
                    "amount": 1800,
                    "status": "pending",
                },
            ],
        },
        "ORD-MM-1008": {
            "order_id": "ORD-MM-1008",
            "customer_email": "frank@example.com",
            "category": "subscription",
            "product": "MegaMart Plus Monthly",
            "amount": 499,
            "total_amount": 499,
            "status": "active",
            "order_status": "active",
            "order_date": "2026-08-01",
            "delivery_date": None,
            "tracking_number": None,
            "shipping_carrier": None,
            "items": [
                {
                    "item_id": "ITM-1008-1",
                    "product": "MegaMart Plus Monthly",
                    "category": "subscription",
                    "amount": 499,
                    "status": "active",
                },
            ],
        },
        "ORD-MM-1009": {
            "order_id": "ORD-MM-1009",
            "customer_email": "alice@example.com",
            "category": "electronics",
            "product": "Wireless Mouse + Cotton Polo",
            "amount": 5300,
            "total_amount": 5300,
            "status": "delivered",
            "order_status": "delivered",
            "order_date": "2026-09-18",
            "delivery_date": "2026-09-26",
            "tracking_number": "TRK-MM-100901",
            "shipping_carrier": "Delhivery",
            "items": [
                {
                    "item_id": "ITM-1009-1",
                    "product": "Wireless Mouse",
                    "category": "electronics",
                    "amount": 2800,
                    "status": "delivered",
                },
                {
                    "item_id": "ITM-1009-2",
                    "product": "Cotton Polo",
                    "category": "clothing",
                    "amount": 1500,
                    "status": "delivered",
                },
                {
                    "item_id": "ITM-1009-3",
                    "product": "Banana Bunch",
                    "category": "perishables",
                    "amount": 1000,
                    "status": "delivered",
                },
            ],
        },
        "ORD-MM-1010": {
            "order_id": "ORD-MM-1010",
            "customer_email": "bob@example.com",
            "category": "electronics",
            "product": "Laptop Stand",
            "amount": 3200,
            "total_amount": 3200,
            "status": "delivered",
            "order_status": "delivered",
            "order_date": "2026-07-01",
            "delivery_date": "2026-07-10",
            "tracking_number": "TRK-MM-101001",
            "shipping_carrier": "DTDC",
            "items": [
                {
                    "item_id": "ITM-1010-1",
                    "product": "Laptop Stand",
                    "category": "electronics",
                    "amount": 3200,
                    "status": "delivered",
                },
            ],
        },
    },
    "tiny-candles": {
        "ORD-TC-2001": {
            "order_id": "ORD-TC-2001",
            "customer_email": "grace@example.com",
            "category": "candles",
            "product": "Lavender Soy Candle",
            "amount": 1450,
            "total_amount": 1450,
            "status": "delivered",
            "order_status": "delivered",
            "order_date": "2026-09-20",
            "delivery_date": "2026-09-25",
            "tracking_number": "TRK-TC-200101",
            "shipping_carrier": "India Post",
            "items": [
                {
                    "item_id": "ITM-2001-1",
                    "product": "Lavender Soy Candle",
                    "category": "candles",
                    "amount": 1450,
                    "status": "delivered",
                },
            ],
        },
        "ORD-TC-2002": {
            "order_id": "ORD-TC-2002",
            "customer_email": "henry@example.com",
            "category": "candles",
            "product": "Candle Gift Set",
            "amount": 3200,
            "total_amount": 3200,
            "status": "delivered",
            "order_status": "delivered",
            "order_date": "2026-08-20",
            "delivery_date": "2026-09-01",
            "tracking_number": "TRK-TC-200201",
            "shipping_carrier": "India Post",
            "items": [
                {
                    "item_id": "ITM-2002-1",
                    "product": "Candle Gift Set",
                    "category": "candles",
                    "amount": 3200,
                    "status": "delivered",
                },
            ],
        },
        "ORD-TC-2003": {
            "order_id": "ORD-TC-2003",
            "customer_email": "iris@example.com",
            "category": "subscription_box",
            "product": "Monthly Candle Box",
            "amount": 999,
            "total_amount": 999,
            "status": "active",
            "order_status": "active",
            "order_date": "2026-09-20",
            "delivery_date": None,
            "tracking_number": None,
            "shipping_carrier": None,
            "items": [
                {
                    "item_id": "ITM-2003-1",
                    "product": "Monthly Candle Box",
                    "category": "subscription_box",
                    "amount": 999,
                    "status": "active",
                },
            ],
        },
        "ORD-TC-2004": {
            "order_id": "ORD-TC-2004",
            "customer_email": "jack@example.com",
            "category": "candles",
            "product": "Vanilla Bean Candle",
            "amount": 750,
            "total_amount": 750,
            "status": "processing",
            "order_status": "processing",
            "order_date": "2026-10-03",
            "delivery_date": None,
            "tracking_number": None,
            "shipping_carrier": None,
            "items": [
                {
                    "item_id": "ITM-2004-1",
                    "product": "Vanilla Bean Candle",
                    "category": "candles",
                    "amount": 750,
                    "status": "processing",
                },
            ],
        },
        "ORD-TC-2005": {
            "order_id": "ORD-TC-2005",
            "customer_email": "grace@example.com",
            "category": "candles",
            "product": "Citrus Burst Candle",
            "amount": 2100,
            "total_amount": 2100,
            "status": "delivered",
            "order_status": "delivered",
            "order_date": "2026-09-22",
            "delivery_date": "2026-09-28",
            "tracking_number": "TRK-TC-200501",
            "shipping_carrier": "India Post",
            "items": [
                {
                    "item_id": "ITM-2005-1",
                    "product": "Citrus Burst Candle",
                    "category": "candles",
                    "amount": 2100,
                    "status": "delivered",
                },
            ],
        },
        "ORD-TC-2006": {
            "order_id": "ORD-TC-2006",
            "customer_email": "henry@example.com",
            "category": "candles",
            "product": "Rose Petal Candle",
            "amount": 1800,
            "total_amount": 1800,
            "status": "shipped",
            "order_status": "shipped",
            "order_date": "2026-10-01",
            "delivery_date": None,
            "tracking_number": "TRK-TC-200601",
            "shipping_carrier": "Delhivery",
            "items": [
                {
                    "item_id": "ITM-2006-1",
                    "product": "Rose Petal Candle",
                    "category": "candles",
                    "amount": 1800,
                    "status": "shipped",
                },
            ],
        },
        "ORD-TC-2007": {
            "order_id": "ORD-TC-2007",
            "customer_email": "iris@example.com",
            "category": "candles",
            "product": "Jasmine Night Candle",
            "amount": 950,
            "total_amount": 950,
            "status": "pending",
            "order_status": "pending",
            "order_date": "2026-10-04",
            "delivery_date": None,
            "tracking_number": None,
            "shipping_carrier": None,
            "items": [
                {
                    "item_id": "ITM-2007-1",
                    "product": "Jasmine Night Candle",
                    "category": "candles",
                    "amount": 950,
                    "status": "pending",
                },
            ],
        },
        "ORD-TC-2008": {
            "order_id": "ORD-TC-2008",
            "customer_email": "jack@example.com",
            "category": "subscription_box",
            "product": "Monthly Candle Box",
            "amount": 999,
            "total_amount": 999,
            "status": "active",
            "order_status": "active",
            "order_date": "2026-06-01",
            "delivery_date": None,
            "tracking_number": None,
            "shipping_carrier": None,
            "items": [
                {
                    "item_id": "ITM-2008-1",
                    "product": "Monthly Candle Box",
                    "category": "subscription_box",
                    "amount": 999,
                    "status": "active",
                },
            ],
        },
        "ORD-TC-2009": {
            "order_id": "ORD-TC-2009",
            "customer_email": "grace@example.com",
            "category": "candles",
            "product": "Candle + Holder Set",
            "amount": 2850,
            "total_amount": 2850,
            "status": "delivered",
            "order_status": "delivered",
            "order_date": "2026-09-24",
            "delivery_date": "2026-09-30",
            "tracking_number": "TRK-TC-200901",
            "shipping_carrier": "India Post",
            "items": [
                {
                    "item_id": "ITM-2009-1",
                    "product": "Sandalwood Candle",
                    "category": "candles",
                    "amount": 1350,
                    "status": "delivered",
                },
                {
                    "item_id": "ITM-2009-2",
                    "product": "Ceramic Candle Holder",
                    "category": "accessories",
                    "amount": 1500,
                    "status": "delivered",
                },
            ],
        },
    },
}

MOCK_REFUND_HISTORY: dict[str, dict[str, list[dict]]] = {
    "megamart": {
        "alice@example.com": [
            {"order_id": "ORD-MM-9001", "amount": 3000, "date": "2026-09-20"},
            {"order_id": "ORD-MM-9002", "amount": 2500, "date": "2026-09-25"},
            {"order_id": "ORD-MM-9003", "amount": 1500, "date": "2026-09-28"},
        ],
    },
    "tiny-candles": {},
}


def get_order(tenant_id: str, order_id: str) -> dict | None:
    return MOCK_ORDERS.get(tenant_id, {}).get(order_id)


def get_order_item(tenant_id: str, order_id: str, item_id: str) -> dict | None:
    order = get_order(tenant_id, order_id)
    if not order:
        return None
    for item in order.get("items", []):
        if item["item_id"] == item_id:
            return item
    return None


def get_tenant_info(tenant_id: str) -> dict | None:
    return MOCK_TENANTS.get(tenant_id)


def get_refund_count_30d(
    tenant_id: str, customer_email: str, today: date | None = None
) -> int:
    today = today or date.today()
    cutoff = today - timedelta(days=30)
    history = MOCK_REFUND_HISTORY.get(tenant_id, {}).get(customer_email, [])
    return sum(1 for r in history if date.fromisoformat(r["date"]) >= cutoff)
