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
from .models import AgentLog
from .models import Conversation
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage
from langchain.agents.middleware import wrap_tool_call


CURRENT_CONVERSATION_ID = None

def log_event(conversation_id,event_type,message):
    
    conversation = Conversation.objects.get(id= conversation_id)

    AgentLog.objects.create(
        conversation=conversation,
        event_type=event_type,
        message = str(message)
    )

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
    return run_manager_agent(case_summary, CURRENT_CONVERSATION_ID)

@tool
def assess_fraud_risk(user_id: int) -> str:
    """
    Consult the risk agent to assess fraud risk for a customer. 
    Use this when refund request looks suspicious or customer has multiple refund requests. 
    Pass the user_id to get a risk verdict.

    """
    log_event(CURRENT_CONVERSATION_ID, "manager", f"Consulting Risk Agent for fraud assessment of user {user_id}.")
    return run_risk_agent(user_id)





def run_support_agent(user_message, conversation_id, order_id, user_id):

    global CURRENT_CONVERSATION_ID
    CURRENT_CONVERSATION_ID = conversation_id
    
    conv = Conversation.objects.get(id=conversation_id)

    # conversation_messages = []
    
    # for msg in conv.messages.order_by("created_at"):
    #     conversation_messages.append({
    #         "role":msg.role,
    #         "content":msg.content
    #     })



    config = {"configurable": {"thread_id": str(conversation_id)}}

    contextual_message = f"[Context: This conversation is about Order #{order_id}, user:{user_id}] {user_message}"

    @wrap_tool_call
    def log_tool_calls(request, handler):
        
        # Before execution
        print("about to tool call")
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call["args"]

        # log tool call
        AgentLog.objects.create(conversation = conv, event_type= "tool_call", message=f"Calling tool {tool_name} with {tool_args}")

        print("tool_name==>", tool_name)
        print("tool_args==>", tool_args) 
        
        result = handler(request)
        print("result ==>", result)
        
        # after execution
        print("finshed tool call")
        AgentLog.objects.create(conversation = conv, event_type= "tool_result", message=f"{tool_name} returned: {str(result)[:200]}")
        return result

    support_agent = create_agent(
                model = llm,
                tools= [get_order_details, get_refund_history, check_delivery_status, escalate_to_manager],
                system_prompt=support_system_prompt,
                checkpointer=InMemorySaver(), # used by LangGraph/LangChain agents to remember the conversation state between messages.
                middleware=[ log_tool_calls ]
            )

    log_event(conversation_id, event_type="support",message=f"Customer: {user_message}")

    result = support_agent.invoke(
        {"messages": [{"role": "user", "content": contextual_message}]},
        config=config,
    )

    # Add these lines
    # from pprint import pprint
    # pprint(result["messages"])

    print("result==>", result["messages"][-1].content)
    final_result = result["messages"][-1].content
 
    # log the final result
    log_event(conversation_id, "final", final_result)

    return final_result


manager_system_prompt = """
You are a Senior Support Manager at CoolBreeze AC.

A customer support agent has escalated a refund request to you.

Your role is to make the FINAL business decision on refund requests.

--------------------------------------------------
Responsibilities
--------------------------------------------------

- Review the complete case summary provided by the support agent.
- Review all available order information.
- Review the customer's refund history.
- If a Risk Agent assessment is provided, use it as part of your decision.
- Make the final refund decision.

--------------------------------------------------
Refund Decision Workflow (MANDATORY)
--------------------------------------------------

1. Carefully review the case summary.

2. If the case contains ANY of the following:

- multiple refund requests
- previous refund denials
- repeated refund requests for the same order
- suspicious customer behaviour
- possible fraud indicators

You MUST call the assess_fraud_risk(user_id) tool.

3. Never make assumptions about fraud yourself.

4. Wait until the Risk Agent returns its assessment.

5. Use BOTH:
   - the case summary
   - the Risk Agent's assessment

to make the final business decision.

6. If a Risk Agent assessment has already been provided, DO NOT call the tool again.

--------------------------------------------------
Tool Usage Rules
--------------------------------------------------

The only tool available for fraud assessment is:

assess_fraud_risk(user_id)

Rules:

- Always use this tool for suspicious refund requests.
- Never skip this tool because you think you already have enough information.
- Never estimate fraud risk yourself.
- Wait for the tool result before making the final decision.

--------------------------------------------------
Decision Rules
--------------------------------------------------

Base every decision ONLY on the facts provided.

Never invent:

- company policies
- refund windows
- warranty periods
- delivery information
- product defects
- customer history

Never assume facts that are not present.

--------------------------------------------------
Decision Format
--------------------------------------------------

Always respond in the following format:

Refund Decision: Approve
or
Refund Decision: Deny

Reason:
<clear explanation>

--------------------------------------------------
Valid Reasons
--------------------------------------------------

Use only facts from the case, such as:

- order age
- delivery status
- order status
- product condition
- refund history
- Risk Agent recommendation

Prioritize the strongest business reason first.

--------------------------------------------------
Customer-Friendly Language
--------------------------------------------------

Write the reason so it can be shown directly to the customer.

Do NOT mention:

- fraud score
- refund ratio
- internal investigations
- internal risk assessment
- internal business rules
- internal decision-making process

Instead, explain the decision in simple, professional language.

Never tell the customer that they are considered "high risk",
"suspicious", or "fraudulent".

Never mention that the Risk Agent was consulted.

Instead, explain the outcome using customer-visible facts such as:
- previous refund decisions
- order history
- delivery status
- product condition
- refund eligibility

The customer should never know that an internal fraud assessment took place.

Example:

Refund Decision: Deny

Reason:
This order has already been reviewed for a refund previously, and after reviewing the order history and account information, we are unable to approve another refund request for the same issue.

--------------------------------------------------
Final Rule
--------------------------------------------------

You are responsible for the final refund decision.

The Risk Agent provides analysis only.

Do not delegate the final decision to any other agent.
"""


manager_agent = create_agent(
    model=llm,
    system_prompt=manager_system_prompt,
    tools=[ assess_fraud_risk ]
)


def run_manager_agent(case_summary, conversation_id):

    log_event(conversation_id, "manager", "Manager started reviewing the refund request.")
    log_event(conversation_id, event_type="manager", message=f"Case Summary:\n{case_summary[:200]}")

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

    log_event(conversation_id, "manager", f"Manager Decision:\n{decision}")

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

    log_event(CURRENT_CONVERSATION_ID, "risk", f"Risk assessment started for user {user_id}.")

    content = f"Please assess the fraud risk for user ID {user_id}. Use your tool to get their profile and return a verdict."
    response = risk_agent.invoke({
        "messages": [{"role": "user", "content": content}]
    })

    for msg in response["messages"]:
        if isinstance(msg, AIMessage):
            for tool in msg.tool_calls:
                log_event(
                CURRENT_CONVERSATION_ID,
                "tool_call",
                f"[Risk Agent] Tool: {tool['name']}\nArguments: {tool['args']}"
            )
                
    for msg in response["messages"]:
        if isinstance(msg, ToolMessage):
            log_event(
            CURRENT_CONVERSATION_ID,
            "tool_result",
            f"[Risk Agent] Tool: {msg.name}\nResult:\n{msg.content}"
        )

    decision = response["messages"][-1].content

    log_event(CURRENT_CONVERSATION_ID, "risk", f"Verdict:{decision[:200]}")
    print("Decision==>",decision)
    return decision
    






