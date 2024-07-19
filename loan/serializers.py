
from rest_framework import serializers

# class LoanPredictionSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = LoanPrediction
#         fields = ['id', 'gender', 'married', 'dependents', 'education', 'self_employed', 'applicant_income',
#                   'co_applicant_income', 'loan_amount', 'loan_amount_term', 'credit_history', 'property_area', 'loan_status']


class LoanPredictionSerializer(serializers.Serializer):
    gender = serializers.CharField(max_length=10)
    married = serializers.CharField(max_length=10)
    dependents = serializers.IntegerField()
    education = serializers.CharField(max_length=50)
    self_employed = serializers.CharField(max_length=10)
    applicant_income = serializers.FloatField()
    co_applicant_income = serializers.FloatField()
    loan_amount = serializers.FloatField()
    loan_amount_term = serializers.FloatField()
    credit_history = serializers.IntegerField()
    property_area = serializers.CharField(max_length=50)
    total_income =serializers.FloatField()
    balance_income = serializers.FloatField()
    emi = serializers.FloatField()
