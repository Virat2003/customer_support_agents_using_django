from django.shortcuts import render, get_object_or_404
import json
from django.http import JsonResponse
import time
from .models import Messages, Conversation
from orders.models import Order
from .agents import run_support_agent
from django.contrib.admin.views.decorators import staff_member_required

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


@staff_member_required
def dashboard(request):
    
    conversations = Conversation.objects.all().order_by("-created_at")

    context = {
        "conversations":conversations
    }

    return render(request, "support/dashboard.html", context)


def conversation_deatil(request,conversation_id):
    
    conversation = get_object_or_404(Conversation, id=conversation_id)
    messages = conversation.messages.order_by("created_at")
    agentlogs = conversation.agentlogs.order_by("created_at")

    context = {
        "conversation":conversation,
        "messages":messages,
        "agentlogs":agentlogs
    }

    return render(request, "support/conversation_detail.html", context)