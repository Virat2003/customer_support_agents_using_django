from langchain_mistralai import ChatMistralAI
from django.conf import settings
from .tools import get_order_details, get_refund_history, check_delivery_status
# from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from .models import Conversation, Messages
from langchain.tools import tool
from pprint import pprint



llm = ChatMistralAI(model=settings.MISTRAL_MODEL, api_key=settings.MISTRAL_API_KEY)

support_system_prompt = """
                You are Maya, a customer support agent at CoolBreeze AC.
                You help customers with issues related to their AC orders.

                Your responsibilities:
                - Always use your tools to gather facts before responding.
                - Check order details when the customer mentions their order.
                - Check refund history before making any refund decisions.
                - Be empathetic but honest.

                Your personality:
                - Friendly and professional.
                - Patient even when the customer is angry.
                - Clear and concise.
                - No emojis.

                Important rules:
                - Always check order details first before responding.
                - Never approve or deny a refund yourself.
                - If a refund decision is needed, tell the customer you are checking with your team.
                - Never use markdown, bold text, or bullet points in your responses.
                - Keep replies conversational and under four sentences.
                """
 
# i want refund , its been many days i am waiting


@tool
def escalate_to_manager(case_summary:str) -> dict:
    """
    REQUIRED TOOL.

    Call this tool whenever a customer requests a refund,
    refund approval,
    refund eligibility,
    or complains about a delayed order and wants a refund.
    Always include customer's user_id in the case summary so manager can assess fraud risk accurately.
    This tool returns the manager's final refund decision.
    Never answer a refund request without calling this tool.
    """
    return run_manager_agent(case_summary)

support_agent = create_agent(
            model = llm,
            tools= [get_order_details, get_refund_history, check_delivery_status, escalate_to_manager],
            system_prompt=support_system_prompt,
            checkpointer=InMemorySaver() # used by LangGraph/LangChain agents to remember the conversation state between messages.
        )


# def support_agent():

def run_support_agent(user_message, conversation_id, order_id, user_id):
    
    # conv = Conversation.objects.get(id=conversation_id)

    # conversation_messages = []
    
    # for msg in conv.messages.order_by("created_at"):
    #     conversation_messages.append({
    #         "role":msg.role,
    #         "content":msg.content
    #     })



    config = {"configurable": {"thread_id": str(conversation_id)}}

    contextual_message = f"[Context: This conversation is about Order #{order_id}, user:{user_id}] {user_message}"

    result = support_agent.invoke(
        {"messages": [{"role": "user", "content": contextual_message}]},
        config=config,
    )

    print("result==>", result["messages"][-1].content)
    pprint(result)
    final_result = result["messages"][-1].content
    return final_result


manager_system_prompt = """
You are a senior support manager at CoolBreeze AC.
A support agent has escalated a customer case to you for a refund decision.

Your responsibilities:
- Review the case summary carefully
- Consider the customer's refund history
- Make a fair and final refund decision
- Give a clear reason for your decision

Your decision options:
- Approve refund — if the case is genuine and within policy
- Deny refund — if the case is suspicious or outside policy
- Escalate to risk team — if you suspect fraud

Important rules:
- Be fair but firm
- Base decision on facts — not emotions
- Always give a specific reason for your decision
- Keep your response concise and professional
"""


# RISK_SYSTEM_PROMPT = """
# You are a fraud risk analyst at CoolBreeze AC.
# A support manager has sent you a customer profile for risk assessment.

# Your job:
# - Analyse the customer's order and refund patterns
# - Identify suspicious behaviour
# - Return a clear risk verdict

# Risk levels:
# - LOW — genuine customer, normal behaviour
# - MEDIUM — some suspicious signals, proceed with caution
# - HIGH — clear fraud pattern, recommend denial

# Your response format:
# - Risk Level: LOW / MEDIUM / HIGH
# - Key Signals: what you found suspicious or genuine
# - Recommendation: what manager should do

# Important:
# - Be objective — base verdict on data only
# - One bad refund does not make someone fraudulent
# - Look for patterns — not isolated incidents
# """

manager_agent = create_agent(
    model=llm,
    system_prompt=manager_system_prompt,
    tools=[]
)


def run_manager_agent(case_summary):
    result = manager_agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": case_summary
        }
    ]
})
    
    print("case_summary==>",case_summary)

    decision = result["messages"][-1].content
    print("Decision==>",decision)
    return decision


    
