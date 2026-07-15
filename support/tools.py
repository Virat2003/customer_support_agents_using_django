from datetime import timedelta

from orders.models import Order, RefundRequest
from django.utils import timezone
from langchain.tools import tool
from .tracking_data import DELIVERY_DATA


@tool
def get_order_details(order_id:int) -> dict:
    
    """Fetch complete order details including status, carrier, tracking number and days since order was placed. 
    Use this when customer mentions their order or complains about delivery."""

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
    

@tool
def get_refund_history(user_id:int) -> dict :

    """Get complete refund history for a user. 
    Use this before making any refund related decisions."""

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

@tool
def check_delivery_status(tracking_number:str, carrier:str) -> dict:

    """ Check current delivery status using tracking number and carrier. 
    Use this when customer complains about delayed or missing delivery."""

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

@tool
def get_customer_risk_profile(user_id:int) -> dict:
    """
    Get complete risk profile for a customer including order history, refund patterns and ratio. 
    Use this to assess fraud risk.
    """
    refunds = RefundRequest.objects.filter(user_id=user_id)
    order = Order.objects.filter(user_id=user_id)

    # recent 90 refund request
    recent_refunds = refunds.filter(created_at__gte= timezone.now()- timedelta(days=90)).count()

    denied = refunds.filter(status="denied").count()
    approved = refunds.filter(status="approved").count()
    pending = refunds.filter(status="pending").count()

    total_refunds = refunds.count()
    total_orders = order.count()

    if total_orders > 0:
        refund_to_oder_ratio = round(total_refunds / total_orders, 2)
    else:
        refund_to_oder_ratio = 0
    
    return {
        "user_id":user_id,
        "total_orders":total_orders,
        "total_refunds":total_refunds,
        "refunds_last_90_days":recent_refunds,
        "denied_refunds":denied,
        "approved_refunds":approved,
        "pending_refunds":pending,
        "refund_to_oder_ratio":refund_to_oder_ratio
    }

    
    
