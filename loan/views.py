# from django.shortcuts import render
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from loan.serializers import LoanPredictionSerializer
# from rest_framework.exceptions import ValidationError
# from loan.models import LoanPrediction
# from django.shortcuts import get_object_or_404

# import numpy as np
# import joblib
# import os


from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from loan.serializers import LoanPredictionSerializer
from rest_framework.exceptions import ValidationError
from loan.models import LoanPrediction
from django.shortcuts import get_object_or_404

import numpy as np
import joblib
import os


class LoanPredictionAPIViews(APIView):
    # Define the list of models
    model_paths = [
        os.path.join('loan', 'trained_model', 'svm_model1.joblib'),
    ]
    models = [joblib.load(model_path) for model_path in model_paths]

    def post(self, request, format=None):
        serializer = LoanPredictionSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data  # Use validated_data instead of data

            # Convert categorical variables to numerical representations
            gender_mapping = {'male': 1, 'female': 0}
            married_mapping = {'no': 0, 'yes': 1}
            self_employed_mapping = {'no': 0, 'yes': 1}
            education_mapping={'Graduate':0,'Not Graduate' :1}
            property_area_mapping={'urban':2,'rural':0,'semiurban':1}
            loan_status={'yes':1,'no':0}

            gender = gender_mapping.get(data['gender'].lower())  # Assuming gender is lowercase
            married = married_mapping.get(data['married'].lower())
            self_employed = self_employed_mapping.get(data['self_employed'].lower())
            education =education_mapping.get(data['education'].lower())
            property_area=property_area_mapping.get(data['property_area'].lower())

            # Convert the data to the appropriate format for prediction
            input_data = np.array([
                gender, married,self_employed, education,property_area,data['dependents'],
               
            #    data[' self_employed'],
                 data['applicant_income'], data['co_applicant_income'],
                data['loan_amount'], data['loan_amount_term'], data['credit_history'],
                # data['property_area'], 
                data['total_income'], data['balance_income'], data['emi']
            ]).reshape(1, -1)

            # Make predictions from all models and aggregate results
            predictions = []
            for model in self.models:
                prediction = model.predict(input_data)
                predictions.append(prediction[0])  # Store the prediction

            # Example: Majority vote for classification
            loan_status = max(set(predictions), key=predictions.count)

            # Convert loan_status back to object (string) form
            loan_status_str = 'approved' if loan_status == 1 else 'not approved'

            return Response({
                "success": True,
                "message": "Prediction successful",
                "status": loan_status_str  # Return loan_status as an object (string)
            }, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# class LoanPredictionAPIViews(APIView):
#     # Define the list of models
#     model_paths = [
#         os.path.join('loan', 'trained_model', 'svm_model1.joblib'),
#     ]
#     models = [joblib.load(model_path) for model_path in model_paths]

#     def post(self, request, format=None):
#         serializer = LoanPredictionSerializer(data=request.data)
#         if serializer.is_valid():
#             data = serializer.data
#             # Convert the data to the appropriate format for prediction
#             input_data = np.array([
#                 data['gender'], data['married'], data['dependents'],
#                 data['education'], data['self_employed'], data['applicant_income'],
#                 data['co_applicant_income'], data['loan_amount'], data['loan_amount_term'],
#                 data['credit_history'], data['property_area'],data['total_income'],data['balance_income'],data['emi']
#             ]).reshape(1, -1)

#             # Make predictions from all models and aggregate results
#             predictions = []
#             for model in self.models:
#                 prediction = model.predict(input_data)
#                 predictions.append(prediction[0])  # Store the prediction

#             # Example: Majority vote for classification
#             loan_status = max(set(predictions), key=predictions.count)

#             return Response({
#                 "success": True,
#                 "message": "Prediction successful",
#                 "status": loan_status
#             }, status=status.HTTP_200_OK)

#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# class LoanPredictionModelAPIViews(APIView):
#     # Define the list of models
#     model_paths = [
#         # os.path.join('loan', 'trained_model', 'dtc_model1.joblib'),
#         os.path.join('loan', 'trained_model', 'gscv_model1.joblib'),
#         # os.path.join('loan', 'trained_model', 'lr_model1.joblib'),
#         # os.path.join('loan', 'trained_model', 'random_forest_model1.joblib'),
#         # os.path.join('loan', 'trained_model', 'svm_model1.joblib'),
#     ]
#     models = [joblib.load(model_path) for model_path in model_paths]

#     def post(self, request, format=None):
#         serializer = LoanPredictionSerializer(data=request.data)
#         if serializer.is_valid():
#             data = serializer.data
#             # Convert the data to the appropriate format for prediction
#             input_data = np.array([
#                 data['gender'], data['married'], data['dependents'],
#                 data['education'], data['self_employed'], data['applicant_income'],
#                 data['co_applicant_income'], data['loan_amount'], data['loan_amount_term'],
#                 data['credit_history'], data['property_area']
#             ]).reshape(1, -1)

#             # Make predictions from all models and aggregate results
#             predictions = []
#             for model in self.models:
#                 prediction = model.predict(input_data)
#                 predictions.append(prediction[0])  # Store the prediction

#             # Example: Majority vote for classification
#             loan_status = max(set(predictions), key=predictions.count)

#             return Response({
#                 "success": True,
#                 "message": "Prediction successful",
#                 "loan_status": loan_status
#             }, status=status.HTTP_200_OK)

#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoanPredictionListAPIViews(APIView):
    '''LIST ALL Loan prediction LIST'''

    def get(self, request, format=None):
        loan_prediction = LoanPrediction.objects.all()  # listing all loan_predicts
        serializers = LoanPredictionSerializer(loan_prediction, many=True)
        return Response({
            "success": True,
            "message": "List All Loan Prediction List",
            "result": serializers.data
        }, status=status.HTTP_201_CREATED)


class LoanPredictionCreateAPIViews(APIView):
    def post(self, request, format=None):
        serializer = LoanPredictionSerializer(data=request.data)
        try:
            if serializer.is_valid(raise_exception=True):
                loan_predict = serializer.save()
                loan_predict_data = {
                    "id": loan_predict.id,
                    "gender": loan_predict.gender,
                    "married": loan_predict.married,
                    "dependents": loan_predict.dependents,
                    "education": loan_predict.education,
                    "self_employed": loan_predict.self_employed,
                    "applicant_income": loan_predict.applicant_income,
                    "co_applicant_income": loan_predict.co_applicant_income,
                    "loan_amount": loan_predict.loan_amount,
                    "loan_amount_term": loan_predict.loan_amount_term,
                    "credit_history": loan_predict.credit_history,
                    "property_area": loan_predict.property_area,
                    "loan_status": loan_predict.loan_status,
                    # "created": loan_predict.created,

                }
                return Response({
                    "success": True,
                    "message": "Successfully loan prediction data created",
                    "result": loan_predict_data
                }, status=status.HTTP_200_OK)

        except ValidationError as e:
            return Response({
                "success": False,
                "message": serializer.errors,

            }, status=status.HTTP_400_BAD_REQUEST)


# class LoanPredictionDetailAPIViews(APIView):
#     def get(self, request, pk):
#         subject = get_object_or_404(LoanPrediction, pk=pk)
#         serializer = LoanPredictionSerializer(subject)
        # return Response(serializer.data)


class LoanPredictionUpdateAPIViews(APIView):
    def put(self, request, pk, format=None):
        try:
            loan_predict = LoanPrediction.objects.get(pk=pk)
        except LoanPrediction.DoesNotExist:
            return Response({
                "success": False,
                "message": "loan prediction data not found."
            }, status=status.HTTP_404_NOT_FOUND)

        serializer = LoanPredictionSerializer(
            loan_predict_data, data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            loan_predict_data = {
                "id": loan_predict.id,
                "gender": loan_predict.gender,
                "married": loan_predict.married,
                "dependents": loan_predict.dependents,
                "education": loan_predict.education,
                "self_employed": loan_predict.self_employed,
                "applicant_income": loan_predict.applicant_income,
                "co_applicant_income": loan_predict.co_applicant_income,
                "loan_amount": loan_predict.loan_amount,
                "loan_amount_term": loan_predict.loan_amount_term,
                "credit_history": loan_predict.credit_history,
                "property_area": loan_predict.property_area,
                "loan_status": loan_predict.loan_status,
                # "created": loan_predict.created,
            }
            return Response({
                "success": True,
                "message": "Successfully loan prediction data  updated ",
                "result": loan_predict_data
            }, status=status.HTTP_200_OK)


class LoanPredictionDeleteAPIViews(APIView):
    def delete(self, request, pk, format=None):

        try:
            loan_predict = LoanPrediction.objects.get(pk=pk)
        except LoanPrediction.DoesNotExist:

            return Response({
                "success": False,
                "message": "Loan predict data  not found."
            }, status=status.HTTP_404_NOT_FOUND)

        loan_predict.delete()

        return Response({
            "success": True,
            "message": "Loan prediction data deleted successfully."
        }, status=status.HTTP_204_NO_CONTENT)


# class LoanPredictionAPIViews(APIView):
#     # Define the list of models
#     model_paths = [
#         os.path.join('loan', 'trained_model', 'svm_model1.joblib'),
#     ]
#     models = [joblib.load(model_path) for model_path in model_paths]

#     def post(self, request, format=None):
#         serializer = LoanPredictionSerializer(data=request.data)
#         if serializer.is_valid():
#             data = serializer.data
#             # Convert the data to the appropriate format for prediction
#             input_data = np.array([
#                 data['gender'], data['married'], data['dependents'],
#                 data['education'], data['self_employed'], data['applicant_income'],
#                 data['co_applicant_income'], data['loan_amount'], data['loan_amount_term'],
#                 data['credit_history'], data['property_area']
#             ]).reshape(1, -1)

#             # Make predictions from all models and aggregate results
#             predictions = []
#             for model in self.models:
#                 prediction = model.predict(input_data)
#                 predictions.append(prediction[0])  # Store the prediction

#             # Example: Majority vote for classification
#             loan_status = max(set(predictions), key=predictions.count)

#             return Response({
#                 "success": True,
#                 "message": "Prediction successful",
#                 "loan_status": loan_status
#             }, status=status.HTTP_200_OK)

#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# # class LoanPredictionModelAPIViews(APIView):
# #     # Define the list of models
# #     model_paths = [
# #         # os.path.join('loan', 'trained_model', 'dtc_model1.joblib'),
# #         os.path.join('loan', 'trained_model', 'gscv_model1.joblib'),
# #         # os.path.join('loan', 'trained_model', 'lr_model1.joblib'),
# #         # os.path.join('loan', 'trained_model', 'random_forest_model1.joblib'),
# #         # os.path.join('loan', 'trained_model', 'svm_model1.joblib'),
# #     ]
# #     models = [joblib.load(model_path) for model_path in model_paths]

# #     def post(self, request, format=None):
# #         serializer = LoanPredictionSerializer(data=request.data)
# #         if serializer.is_valid():
# #             data = serializer.data
# #             # Convert the data to the appropriate format for prediction
# #             input_data = np.array([
# #                 data['gender'], data['married'], data['dependents'],
# #                 data['education'], data['self_employed'], data['applicant_income'],
# #                 data['co_applicant_income'], data['loan_amount'], data['loan_amount_term'],
# #                 data['credit_history'], data['property_area']
# #             ]).reshape(1, -1)

# #             # Make predictions from all models and aggregate results
# #             predictions = []
# #             for model in self.models:
# #                 prediction = model.predict(input_data)
# #                 predictions.append(prediction[0])  # Store the prediction

# #             # Example: Majority vote for classification
# #             loan_status = max(set(predictions), key=predictions.count)

# #             return Response({
# #                 "success": True,
# #                 "message": "Prediction successful",
# #                 "loan_status": loan_status
# #             }, status=status.HTTP_200_OK)

# #         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# class LoanPredictionListAPIViews(APIView):
#     '''LIST ALL Loan prediction LIST'''

#     def get(self, request, format=None):
#         loan_prediction = LoanPrediction.objects.all()  # listing all loan_predicts
#         serializers = LoanPredictionSerializer(loan_prediction, many=True)
#         return Response({
#             "success": True,
#             "message": "List All Loan Prediction List",
#             "result": serializers.data
#         }, status=status.HTTP_201_CREATED)


# class LoanPredictionCreateAPIViews(APIView):
#     def post(self, request, format=None):
#         serializer = LoanPredictionSerializer(data=request.data)
#         try:
#             if serializer.is_valid(raise_exception=True):
#                 loan_predict = serializer.save()
#                 loan_predict_data = {
#                     "id": loan_predict.id,
#                     "gender": loan_predict.gender,
#                     "married": loan_predict.married,
#                     "dependents": loan_predict.dependents,
#                     "education": loan_predict.education,
#                     "self_employed": loan_predict.self_employed,
#                     "applicant_income": loan_predict.applicant_income,
#                     "co_applicant_income": loan_predict.co_applicant_income,
#                     "loan_amount": loan_predict.loan_amount,
#                     "loan_amount_term": loan_predict.loan_amount_term,
#                     "credit_history": loan_predict.credit_history,
#                     "property_area": loan_predict.property_area,
#                     "loan_status": loan_predict.loan_status,
#                     # "created": loan_predict.created,

#                 }
#                 return Response({
#                     "success": True,
#                     "message": "Successfully loan prediction data created",
#                     "result": loan_predict_data
#                 }, status=status.HTTP_200_OK)

#         except ValidationError as e:
#             return Response({
#                 "success": False,
#                 "message": serializer.errors,

#             }, status=status.HTTP_400_BAD_REQUEST)


# # class LoanPredictionDetailAPIViews(APIView):
# #     def get(self, request, pk):
# #         subject = get_object_or_404(LoanPrediction, pk=pk)
# #         serializer = LoanPredictionSerializer(subject)
#         # return Response(serializer.data)


# class LoanPredictionUpdateAPIViews(APIView):
#     def put(self, request, pk, format=None):
#         try:
#             loan_predict = LoanPrediction.objects.get(pk=pk)
#         except LoanPrediction.DoesNotExist:
#             return Response({
#                 "success": False,
#                 "message": "loan prediction data not found."
#             }, status=status.HTTP_404_NOT_FOUND)

#         serializer = LoanPredictionSerializer(
#             loan_predict_data, data=request.data)
#         if serializer.is_valid(raise_exception=True):
#             serializer.save()
#             loan_predict_data = {
#                 "id": loan_predict.id,
#                 "gender": loan_predict.gender,
#                 "married": loan_predict.married,
#                 "dependents": loan_predict.dependents,
#                 "education": loan_predict.education,
#                 "self_employed": loan_predict.self_employed,
#                 "applicant_income": loan_predict.applicant_income,
#                 "co_applicant_income": loan_predict.co_applicant_income,
#                 "loan_amount": loan_predict.loan_amount,
#                 "loan_amount_term": loan_predict.loan_amount_term,
#                 "credit_history": loan_predict.credit_history,
#                 "property_area": loan_predict.property_area,
#                 "loan_status": loan_predict.loan_status,
#                 # "created": loan_predict.created,
#             }
#             return Response({
#                 "success": True,
#                 "message": "Successfully loan prediction data  updated ",
#                 "result": loan_predict_data
#             }, status=status.HTTP_200_OK)


# class LoanPredictionDeleteAPIViews(APIView):
#     def delete(self, request, pk, format=None):

#         try:
#             loan_predict = LoanPrediction.objects.get(pk=pk)
#         except LoanPrediction.DoesNotExist:

#             return Response({
#                 "success": False,
#                 "message": "Loan predict data  not found."
#             }, status=status.HTTP_404_NOT_FOUND)

#         loan_predict.delete()

#         return Response({
#             "success": True,
#             "message": "Loan prediction data deleted successfully."
#         }, status=status.HTTP_204_NO_CONTENT)
