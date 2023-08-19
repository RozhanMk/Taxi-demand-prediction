from rest_framework import serializers


class DatasetSerializer(serializers.Serializer):
    MODEL_CHOICES = (("xgboost", "xgboost"), ("deep", "deep"))
    TIMESTAMP_CHOICES = (("3", "3"))
    ITERATION_CHOICES = ()
    for i in range(1, 9):
        pair = (str(i), str(i))
        ITERATION_CHOICES += (pair,)

    model = serializers.ChoiceField(choices=MODEL_CHOICES)
    timestamp = serializers.ChoiceField(choices=TIMESTAMP_CHOICES)
    iteration = serializers.ChoiceField(choices=ITERATION_CHOICES)
    
    file = serializers.FileField()
