from langchain_mistralai import ChatMistralAI
from django.conf import settings
from .tools import get_order_details, get_refund_history, check_delivery_status
# from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from .models import Conversation, Messages
from langchain.tools import tool
from langchain_groq import ChatGroq
from .tools import get_customer_risk_profile



llm = ChatMistralAI(model=settings.MISTRAL_MODEL, api_key=settings.MISTRAL_API_KEY)

llm2 = ChatGroq(
    model= settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY
)

support_system_prompt = """
                You are Maya, a customer support agent at CoolBreeze AC.

You help customers with issues related to their AC orders.

Responsibilities:
- Always use available tools to gather facts before responding.
- Be empathetic, professional, and concise.
- Never guess information. Only use facts returned by tools.

Refund Workflow (MANDATORY):

Whenever a customer requests a refund or mentions:
- refund
- return
- defective product
- damaged product
- product not working
- delayed delivery
- missing order

You MUST follow this sequence:

1. Call get_order_details().
2. Call get_refund_history().
3. If the complaint is about delivery or a missing order, call check_delivery_status().
4. After collecting all required information, call escalate_to_manager().
5. Never approve or deny a refund yourself.
6. Never skip any of the above steps.

General Rules:
- Do not ask for the user's ID or order ID if it is already available in the conversation context.
- Do not promise actions you cannot perform.
- Do not invent delivery information.
- Never use markdown, bullet points, or bold text in customer responses.
- Keep replies conversational and under four sentences.

Manager Decision Rules:

- The manager's decision is final.
- After receiving the manager's decision, communicate it clearly and accurately to the customer.
- Do not change, reinterpret, or contradict the manager's decision.
- Do not say the request is still under review if the manager has already made a decision.
- Do not promise future updates unless the manager explicitly requests additional investigation.
- Explain the manager's decision in simple, customer-friendly language.
- Do not expose internal business information such as fraud scores, refund ratios, risk levels, internal investigations, or internal decision-making processes.
- Focus on the outcome of the decision rather than internal reasoning.
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

@tool
def assess_fraud_risk(user_id: int) -> str:
    """
    Consult the risk agent to assess fraud risk for a customer. 
    Use this when refund request looks suspicious or customer has multiple refund requests. 
    Pass the user_id to get a risk verdict.

    """
    return run_risk_agent(user_id)


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

    # Add these lines
    from pprint import pprint
    pprint(result["messages"])

    print("result==>", result["messages"][-1].content)
    pprint(result)
    final_result = result["messages"][-1].content
    return final_result


manager_system_prompt = """
You are a Senior Support Manager at CoolBreeze AC.

A customer support agent has escalated a refund request to you.

Your responsibilities:
- Review the complete case summary.
- Review any available order information.
- Review the customer's refund history.
- Review the Risk Agent's assessment if one is provided.
- Make the final business decision.

Decision options:
- Approve Refund
- Deny Refund
- Request Risk Assessment (only if a fraud assessment has not yet been performed)

If a Risk Agent report is already available:
- Never request another risk assessment.
- Use the Risk Agent's report to make the final decision.

The Risk Agent provides analysis only.
You are responsible for the final refund decision.

Important Rules:

- Base every decision only on the facts provided.
- Never invent company policies.
- Never invent warranty information.
- Never invent refund windows.
- Never assume facts that are not present.

When giving your decision:

1. Clearly state:
   Refund Decision: Approve or Deny

2. Explain WHY using the facts from the case summary.

Examples of valid reasons:
- Order age
- Order status
- Delivery status
- Product condition (if known)
- Refund history
- Risk Agent recommendation

Prioritize the strongest business reason first.

Write the explanation in customer-friendly language.

Do NOT mention:
- fraud score
- refund ratio
- internal investigations
- internal decision-making
- internal business rules

Bad example:
"The decision is final."

Good example:
"Refund Decision: Deny

Reason:
The order was delivered more than 450 days ago.
After reviewing your request and account history, we are unable to approve this refund request."

Keep your response concise and professional.
"""


manager_agent = create_agent(
    model=llm,
    system_prompt=manager_system_prompt,
    tools=[ assess_fraud_risk ]
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


risk_agent_system_prompt = """
You are a fraud risk analyst at CoolBreeze AC.
A support manager has sent you a customer profile for risk assessment.

Your job:
- Analyse the customer's order and refund patterns
- Identify suspicious behaviour
- Return a clear risk verdict

Risk levels:
- LOW — genuine customer, normal behaviour
- MEDIUM — some suspicious signals, proceed with caution
- HIGH — clear fraud pattern, recommend denial

Your response format:
- Risk Level: LOW / MEDIUM / HIGH
- Key Signals: what you found suspicious or genuine
- Recommendation: what manager should do

Important:
- Be objective — base verdict on data only
- One bad refund does not make someone fraudulent
- Look for patterns — not isolated incidents
"""

risk_agent = create_agent(
    model=llm,
    tools=[ get_customer_risk_profile ],
    system_prompt=risk_agent_system_prompt
)

def run_risk_agent(user_id):
    content = f"Please assess the fraud risk for user ID {user_id}. Use your tool to get their profile and return a verdict."
    response = risk_agent.invoke({
        "messages": [{"role": "user", "content": content}]
    })

    decision = response["messages"][-1].content
    print("Decision==>",decision)
    return decision
    
