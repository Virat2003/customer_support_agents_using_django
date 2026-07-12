from django.contrib import admin
from .models import Conversation, Messages, AgentLog
# Register your models here.
admin.site.register(Conversation)
admin.site.register(Messages)
admin.site.register(AgentLog)
