import numpy as np
from rest_framework.response import Response
from .serializers import DatasetSerializer
from rest_framework.viewsets import ViewSet
import pandas as pd
from .MLpipline import create_demand
from rest_framework.views import APIView

class PlotAPIView(APIView):
    def get(self, request):
        file_path = 'plot.png'  # Replace with the actual file path
        return Response({'image_url': request.build_absolute_uri(file_path)})

"""
get model name, timestamp, iteration and batch file (from TLC nyc dataset) that contains info from yellow taxi in nyc.
return predicted demand for the furture hours for each LocationID in nyc.
"""
class UploadViewSet(ViewSet):

    ACCEPTED_COLUMNS = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance',
            'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra',
            'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge', 'Airport_fee']
    serializer_class = DatasetSerializer
    

    def list(self, request):
        return Response("WELCOME TO TAXI DEMAND PREDICTION API! Please only use 3 as timestamp. choose 'xgboost' or 'deep' as Model")

    def upload(self, request):
        # Initialize a serializer with incoming data
        serializer = DatasetSerializer(data=self.request.data)
        
        # Check if the serialized data is valid, raise an exception if not
        if serializer.is_valid(raise_exception=True):
            pass

        # Extract relevant data from the validated serializer
        timestamp = serializer.validated_data['timestamp']
        model_name = serializer.validated_data['model']
        iteration = serializer.validated_data['iteration']
        file = serializer.validated_data['file']
        if timestamp != 3:
            return Response("Please only use 3 as timestamp!")
        if iteration <= 0 or iteration >= 9:
            return Response("Please use a number in range 1-8 for iteration!")
        
        # Read a Parquet file (columnar storage format)
        df_input = pd.read_parquet(file)

        # verify data
        columns = df_input.columns.to_list()
        for column in columns:
            if column not in self.ACCEPTED_COLUMNS:
                return Response("dataset should contains columns only from yellow taxi trip records.")
        
        # get xgboost prediction based on input dataset and timestamp
        if model_name == "xgboost" or model_name == "Xgboost":
            predictions, evaluation = create_demand.Prediction(df_input, model_name, timestamp, iteration).predict_xgboost()
            return Response({
                "data": predictions.reset_index().values,
                "evaluation": evaluation
            })
        else:
        # get deep model prediction
            predictions = create_demand.Prediction(df_input, f'{timestamp}h_{model_name}', timestamp, iteration).forecast_deep()
            return Response({
                'meta': {
                    'model': model_name,
                    'timestamp': timestamp
                },
                'data': predictions
            })



        
