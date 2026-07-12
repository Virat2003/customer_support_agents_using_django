from orders.models import Order, RefundRequest
from django.utils import timezone
from langchain.tools import tool
from .tracking_data import DELIVERY_DATA


# @tool
def get_order_details(order_id):
    # """Get the status of an order."""
    try:
        order = Order.objects.get(id= order_id)
        return {
            "order_id":order.id,
            "product_name":order.product_name,
            "amount":str(order.amount),
            "status":order.status,
            "carrier":order.carrier,
            "tracking_number":order.tracking_number,
            "delivery_addreess":order.delivery,
            "ordered_on":order.created_at.strftime("%d %b %Y"),
            "days_since_order":(timezone.now()-order.created_at).days,
        }
    except Order.DoesNotExist:
        return {"error":f"order #{order_id} Not Found."}


# @tool
def get_refund_history(user_id):
    # """Get the refund history of an order."""
    refunds = RefundRequest.objects.filter(user_id=user_id).order_by("-created_at")
    
    history = []
    for r in refunds:
        history.append({
            "order_id":r.order.id,
            "product":r.order.product,
            "reason":r.reason,
            "status":r.status,
            "requested_on":r.created_at.strftime("%d %b %Y"),
        })

    return {
        "total_refund_requests":len(history),
        "history":history
    }


def check_delivery_status(tracking_number, carrier):
    default_response = {
        "status": "Unknown",
        "last_location": "Tracking info unavailable",
        "last_update": "N/A",
        "estimated_delivery": "Contact carrier directly",
        "delay_reason": "No updates from carrier",
    }
    result = DELIVERY_DATA.get(tracking_number, default_response)

    result["tracking_number"] = tracking_number
    result["carrier"] = carrier

    return result


