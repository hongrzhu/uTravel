from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import ChatSession, Message, TravelPlan
from .serializers import ChatSessionSerializer, MessageSerializer, TravelPlanSerializer
from travel_planner.__main__ import compile_graph  # Import the main entry point
from travel_planner.config.settings import SYSTEM_PROMPT
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import logging

class ChatSessionViewSet(viewsets.ModelViewSet):
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer

    @action(detail=False, methods=['post'])
    def start_session(self, request):
        """Start a new chat session"""
        session = ChatSession.objects.create()
        
        # Add system message to initialize the conversation
        Message.objects.create(
            session=session,
            message_type='system',
            content=SYSTEM_PROMPT
        )
        
        # Add initial greeting
        Message.objects.create(
            session=session,
            message_type='agent',
            content="Welcome to uTravel! I'm your AI travel planning assistant. Tell me about your travel wishes! For example, 'I'd like a 3-day adventure in Paris focusing on museums and cafes.'"
        )
        
        serializer = self.get_serializer(session)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'])
    def send_message(self, request, pk=None):
        """Send a message to the agent and get response"""
        session = self.get_object()
        user_message = request.data.get('message')
        
        if not user_message:
            return Response(
                {'error': 'Message content is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        # Create user message
        Message.objects.create(
            session=session,
            message_type='user',
            content=user_message
        )

        # Initialize conversation state as used in travel_planner
        messages = []
        for msg in session.messages.all().order_by('timestamp'):
            if msg.message_type == 'user':
                messages.append(HumanMessage(content=msg.content))
            elif msg.message_type == 'agent':
                messages.append(AIMessage(content=msg.content))
            elif msg.message_type == 'system':
                messages.append(SystemMessage(content=msg.content))

        conversation_state = {
            "messages": messages,
            "current_plan": None,
            "error_message": None
        }

        # Get latest plan if exists
        if session.travel_plans.filter(is_final=True).exists():
            conversation_state["current_plan"] = session.travel_plans.filter(is_final=True).first().itinerary

        # Initialize the travel planner
        app = compile_graph()

        try:
            # Run the travel planner graph
            config = {"recursion_limit": 25}
            graph_output_state = app.invoke(conversation_state, config=config)

            if graph_output_state:
                # Update conversation state
                conversation_state["messages"] = graph_output_state.get("messages", conversation_state["messages"])
                conversation_state["current_plan"] = graph_output_state.get("current_plan", conversation_state["current_plan"])
                conversation_state["error_message"] = graph_output_state.get("error_message")

                # Get the latest AI message
                last_ai_message = next((msg for msg in reversed(conversation_state["messages"]) 
                                     if isinstance(msg, AIMessage)), None)

                if last_ai_message:
                    # Store the AI message
                    agent_message = Message.objects.create(
                        session=session,
                        message_type='agent',
                        content=last_ai_message.content
                    )

                    # Store the plan if one was generated
                    has_plan = False
                    if conversation_state["current_plan"]:
                        TravelPlan.objects.create(
                            session=session,
                            itinerary=conversation_state["current_plan"],
                            is_final=True
                        )
                        has_plan = True

                    return Response({
                        'message': MessageSerializer(agent_message).data,
                        'has_plan': has_plan
                    })
                
        except Exception as e:
            Message.objects.create(
                session=session,
                message_type='system',
                content=f"Error: {str(e)}"
            )
            return Response(
                {'error': f'Failed to process message: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response(
            {'error': 'No response generated'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    @action(detail=True, methods=['get'])
    def get_latest_plan(self, request, pk=None):
        """Get the latest travel plan for the session"""
        session = self.get_object()
        plan = session.travel_plans.filter(is_final=True).last()
        
        if not plan:
            return Response(
                {'error': 'No travel plan found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
            
        return Response(TravelPlanSerializer(plan).data)
