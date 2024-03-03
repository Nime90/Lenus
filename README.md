# Lenus
In this repository i will develop a code to solve a customer conversion analysis for Lenus.

Description
With the data provided in the file customer_data_sample.csv using any method you deem fit, answer
the question:
"What are the most important factors for predicting whether a customer has converted or not?"

Converted customer is represented in the data in the field "converted", and the nature of what this
conversion means is (intentionally) unknown in the context of the challenge.

Fields
| field | explanation |
|---|---|
| customer_id | Numeric id for a customer
| converted | Whether a customer converted to the product (1) or not (0)
| customer_segment | Numeric id of a customer segment the customer belongs to
| gender | Customer gender
| age | Customer age
| related_customers | Numeric - number of people who are related to the customer
| family_size | Numeric - size of family members
| initial_fee_level | Initial services fee level the customer is enrolled to
| credit_account_id | Identifier (hash) for the customer credit account. If customer has none, they are
shown as "9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0"
| branch | Which branch the customer mainly is associated with |

Development:
To complete the task I will first familiarize with the data. Trying to deeply understand it. once it is done I will try to clean and get data ready to the analysis (dropping empty cells and adding necessary dummy columns and summarizations). I will finally perform logistic regressions using a machine learning approach and a "classical" statistical approach.
In the end I will analyze the data and create a presetnation in ppt and tableau to convey the results.

In this repos you will find the notebook containing all the steps i did (including observations and comments) and the python file that can be used for running over and over the logistic regressions changing the independent variables.
Data and PDF file with the description have been left out on purpose (for security reason: I have not been instructed about the privacy of the data received and this is a public repository)