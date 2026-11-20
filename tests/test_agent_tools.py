from __future__ import annotations

import pytest

from app.graph.agent.tools import (
    cancel_order,
    get_order_details,
    get_order_tracking,
    get_return_eligibility,
    get_shipping_label,
    initiate_refund,
    initiate_return,
)


class TestGetOrderDetails:
    def test_existing_order(self):
        result = get_order_details("megamart", "ORD-MM-1001")
        assert result["order_id"] == "ORD-MM-1001"
        assert result["customer_email"] == "alice@example.com"
        assert result["order_status"] == "delivered"
        assert len(result["items"]) >= 1

    def test_multi_item_order(self):
        result = get_order_details("megamart", "ORD-MM-1009")
        assert result["order_id"] == "ORD-MM-1009"
        assert len(result["items"]) == 3

    def test_not_found(self):
        result = get_order_details("megamart", "ORD-FAKE")
        assert "error" in result


class TestGetReturnEligibility:
    def test_eligible_electronics_within_window(self):
        result = get_return_eligibility("megamart", "ORD-MM-1001")
        assert result["category"] == "electronics"

    def test_perishable_not_eligible(self):
        result = get_return_eligibility("megamart", "ORD-MM-1003")
        assert result["eligible"] is False
        assert "not allowed" in result["reason"].lower()

    def test_old_order_outside_window(self):
        result = get_return_eligibility("megamart", "ORD-MM-1010")
        assert result["eligible"] is False
        assert "expired" in result["reason"].lower()

    def test_item_level_eligibility(self):
        result = get_return_eligibility("megamart", "ORD-MM-1009", "ITM-1009-1")
        assert result["category"] == "electronics"

    def test_item_perishable_in_multi_order(self):
        result = get_return_eligibility("megamart", "ORD-MM-1009", "ITM-1009-3")
        assert result["eligible"] is False

    def test_not_found_order(self):
        result = get_return_eligibility("megamart", "ORD-FAKE")
        assert result["eligible"] is False

    def test_not_found_item(self):
        result = get_return_eligibility("megamart", "ORD-MM-1001", "ITM-FAKE")
        assert result["eligible"] is False


class TestInitiateReturn:
    def test_success(self):
        result = initiate_return("megamart", "ORD-MM-1001")
        assert result["success"] is True
        assert "return_id" in result
        assert "Wireless Headphones" in result["product"]

    def test_item_level(self):
        result = initiate_return("megamart", "ORD-MM-1009", "ITM-1009-2")
        assert result["success"] is True
        assert "Cotton Polo" in result["product"]

    def test_not_found(self):
        result = initiate_return("megamart", "ORD-FAKE")
        assert result["success"] is False


class TestInitiateRefund:
    def test_success(self):
        result = initiate_refund("megamart", "ORD-MM-1001", 4500)
        assert result["success"] is True
        assert result["amount"] == 4500
        assert "refund_id" in result

    def test_with_refund_type(self):
        result = initiate_refund("megamart", "ORD-MM-1001", 4500, "store_credit")
        assert result["refund_type"] == "store_credit"

    def test_not_found(self):
        result = initiate_refund("megamart", "ORD-FAKE", 100)
        assert result["success"] is False


class TestGetShippingLabel:
    def test_success(self):
        result = get_shipping_label("megamart", "ORD-MM-1001")
        assert result["success"] is True
        assert "label_url" in result

    def test_not_found(self):
        result = get_shipping_label("megamart", "ORD-FAKE")
        assert result["success"] is False


class TestCancelOrder:
    def test_cancel_pending(self):
        result = cancel_order("megamart", "ORD-MM-1007")
        assert result["success"] is True
        assert "USB-C Hub" in result["product"]

    def test_cancel_processing(self):
        result = cancel_order("megamart", "ORD-MM-1005")
        assert result["success"] is True

    def test_cancel_shipped_fails(self):
        result = cancel_order("megamart", "ORD-MM-1006")
        assert result["success"] is False
        assert "cannot be cancelled" in result["error"].lower()

    def test_cancel_delivered_fails(self):
        result = cancel_order("megamart", "ORD-MM-1001")
        assert result["success"] is False

    def test_not_found(self):
        result = cancel_order("megamart", "ORD-FAKE")
        assert result["success"] is False


class TestGetOrderTracking:
    def test_shipped_has_tracking(self):
        result = get_order_tracking("megamart", "ORD-MM-1006")
        assert result["tracking_number"] is not None
        assert result["carrier"] is not None

    def test_pending_no_tracking(self):
        result = get_order_tracking("megamart", "ORD-MM-1007")
        assert result["tracking_number"] is None
        assert "not been shipped" in result["message"].lower()

    def test_not_found(self):
        result = get_order_tracking("megamart", "ORD-FAKE")
        assert "error" in result
