from django.urls import path, include
from loan import views
app_name = "loan"
urlpatterns = [
    # ============= loan prediction view ==============
    path('new/',
         views.LoanPredictionCreateAPIViews.as_view(), name="loan_create"),  # loan create
    path('list/',
         views.LoanPredictionListAPIViews.as_view(), name="loan_list"),  # loan list view
    path('update/',
         views.LoanPredictionUpdateAPIViews.as_view(), name="loan_update"),  # loan update view
    path('delete/',
         views.LoanPredictionDeleteAPIViews.as_view(), name="loan_delete"),  # loan delete view
    path('predict/',
         views.LoanPredictionAPIViews.as_view(), name='loan_prediction'),
]
