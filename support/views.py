from django.shortcuts import render, get_object_or_404
import json
from django.http import JsonResponse
import time
from .models import Messages, Conversation
from orders.models import Order
from .agents import run_support_agent

# Create your views here.
def chat(request, order_id):
    if request.method == "POST":
        data = json.loads(request.body)
        user_message = data.get("message")
        # print(user_message)

        order = get_object_or_404(Order,id=order_id,user=request.user)

        conversation, created = Conversation.objects.get_or_create(user=request.user,order=order)

        Messages.objects.create(conversation=conversation, role="user", content=user_message)

        reply = run_support_agent(user_message, conversation.id, order.id ,request.user.id)

        Messages.objects.create(conversation=conversation, role="agent", content=reply)

        if not user_message:
            return JsonResponse({"error":"not received"}, status=400)
    # time.sleep(5)
    return JsonResponse({"reply":reply, "message":user_message})
    # return user_message 