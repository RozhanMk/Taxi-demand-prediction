from rest_framework import serializers


class DatasetSerializer(serializers.Serializer):
    model = serializers.CharField()
    timestamp = serializers.IntegerField()
    iteration = serializers.IntegerField()
    file = serializers.FileField()
