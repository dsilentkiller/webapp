from django.db import models

GENDER = [
    ('male', 'Male'),
    ('female', 'Female')
]

EDUCATION = [
    ('graduate', 'Graduate'),
    ('not graduate', 'Not Graduate')
]

PROPERTY_AREA = [
    ('urban', 'Urban'),
    ('semiurban', 'Semiurban'),
    ('rural', 'Rural')
]

LOAN_STATUS = [
    ('yes', 'Yes'),
    ('no', 'No')
]


class LoanPrediction(models.Model):
    gender = models.CharField(choices=GENDER, max_length=100,
                              null=False, blank=False, help_text="Gender is required")
    married = models.BooleanField(null=False)
    dependents = models.CharField(
        max_length=10, null=False, blank=False, help_text="Number of dependents")
    education = models.CharField(
        choices=EDUCATION, max_length=50, null=False, blank=False, help_text="Education is required")
    self_employed = models.BooleanField(default=False)
    applicant_income = models.FloatField(
        null=False, blank=False, help_text="Enter your income amount e.g., 25000 per month")
    co_applicant_income = models.FloatField(
        null=False, blank=False, help_text="Enter your partner's income amount e.g., 20000 per month")
    loan_amount = models.FloatField(
        null=False, blank=False, help_text="Enter your loan amount e.g., 1000000")
    loan_amount_term = models.FloatField(
        null=False, blank=False, help_text="Enter the term of the loan in months e.g., 360")
    credit_history = models.BooleanField(default=True)
    property_area = models.CharField(
        choices=PROPERTY_AREA, max_length=255, blank=False, null=False)
    loan_status = models.CharField(
        choices=LOAN_STATUS, max_length=255, null=True, blank=True)
    total_income =models.FloatField(default=0.0,null=True,blank=True,help_text="Enter your loan amount e.g., 1000000")
    emi =models.FloatField(default=0.0,null=True,blank=True,help_text="Enter your loan amount e.g., 1000000")
    balance_income =models.FloatField(default=0.0,null=True,blank=True,help_text="Enter your loan amount e.g., 1000000")
    
    created = models.DateTimeField(auto_now_add=True, null=True,blank=True)
    updated = models.DateTimeField(null=True,blank=True)

    def __str__(self):
        return f"LoanPrediction({self.id}, {self.gender}, {self.married}, {self.education}, {self.loan_status})"
