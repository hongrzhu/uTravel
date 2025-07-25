from rest_framework import serializers
from .models import ChatSession, Message, TravelPlan

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'message_type', 'content', 'timestamp']

class TravelPlanSerializer(serializers.ModelSerializer):
    class Meta:
        model = TravelPlan
        fields = ['id', 'itinerary', 'created_at', 'updated_at', 'is_final']

class ChatSessionSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)
    travel_plans = TravelPlanSerializer(many=True, read_only=True)

    class Meta:
        model = ChatSession
        fields = ['session_id', 'created_at', 'is_active', 'last_interaction', 'messages', 'travel_plans']
