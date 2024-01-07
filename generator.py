import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget
)
import pandas as pd
from pulp import *
import numpy as np

class NutritionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Food List Generator')
        self.setGeometry(100, 100, 1200, 800)

        # User input labels
        self.kg_label = QLabel("Enter your weight (kg):", self)
        self.calories_label = QLabel("Enter your daily calorie intake:", self)

        # Line edits for user input
        self.kg_input = QLineEdit(self)
        self.calories_input = QLineEdit(self)

        # Button to trigger diet calculation
        self.calculate_button = QPushButton('Generate List', self)
        self.calculate_button.clicked.connect(self.calculate_diet)

        # QTabWidget to organize tables for each day
        self.tab_widget = QTabWidget(self)

        # Labels for displaying daily nutritional values
        self.daily_protein_label = QLabel("Daily Protein (g):", self)
        self.daily_carbs_label = QLabel("Daily Carbohydrates (g):", self)
        self.daily_fat_label = QLabel("Daily Fat (g):", self)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.kg_label)
        layout.addWidget(self.kg_input)
        layout.addWidget(self.calories_label)
        layout.addWidget(self.calories_input)
        layout.addWidget(self.calculate_button)
        layout.addWidget(self.daily_protein_label)
        layout.addWidget(self.daily_carbs_label)
        layout.addWidget(self.daily_fat_label)
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

    def calculate_diet(self):
        # Get user input for weight and daily calories
        kg = float(self.kg_input.text())
        calories = float(self.calories_input.text())

        # Read and preprocess the data
        data = pd.read_csv('nutrition.csv').drop('Unnamed: 0', axis=1)
        data = data[['name', 'serving_size', 'calories', 'carbohydrate', 'total_fat', 'protein']]
        data['carbohydrate'] = np.array([data['carbohydrate'].tolist()[i].split(' ') for i in range(len(data))])[:, 0].astype('float')
        data['protein'] = np.array([data['protein'].tolist()[i].split(' ') for i in range(len(data))])[:, 0].astype('float')
        data['total_fat'] = np.array([data['total_fat'].tolist()[i].split('g') for i in range(len(data))])[:, 0].astype('float')

        # Create the days of the week and the dataset
        week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        split_values = np.linspace(0, len(data), 8).astype(int)
        split_values[-1] = split_values[-1] - 1

        def random_dataset():
            frac_data = data.sample(frac=1).reset_index().drop('index', axis=1)
            day_data = [frac_data.loc[split_values[s]:split_values[s + 1]] for s in range(len(split_values) - 1)]
            return dict(zip(week_days, day_data))

        # Calculate nutritional values
        def build_nutritional_values(kg, calories):
            protein_calories = kg * 4
            carb_calories = calories / 2.0
            fat_calories = calories - carb_calories - protein_calories

            protein_grams = protein_calories / 4.0
            carbs_grams = carb_calories / 4.0
            fat_grams = fat_calories / 9.0

            return {
                'Protein Calories': protein_calories,
                'Carbohydrates Calories': carb_calories,
                'Fat Calories': fat_calories,
                'Protein Grams': protein_grams,
                'Carbohydrates Grams': carbs_grams,
                'Fat Grams': fat_grams
            }

        # Calculate daily nutritional values based on user input for weight and calories
        nutritional_values = build_nutritional_values(kg, calories)

        # Display daily nutritional values in the user interface
        self.daily_protein_label.setText(f"Daily Protein (g): {nutritional_values['Protein Grams']:.2f}")
        self.daily_carbs_label.setText(f"Daily Carbohydrates (g): {nutritional_values['Carbohydrates Grams']:.2f}")
        self.daily_fat_label.setText(f"Daily Fat (g): {nutritional_values['Fat Grams']:.2f}")

        # Extract nutritional values in grams
        def extract_gram(table):
            protein_grams = table['Protein Calories'] / 4.0
            carbs_grams = table['Carbohydrates Calories'] / 4.0
            fat_grams = table['Fat Calories'] / 9.0
            return {'Protein Grams': protein_grams, 'Carbohydrates Grams': carbs_grams, 'Fat Grams': fat_grams}

        # Create models for individual days and the total model
        days_data = random_dataset()

        def model(day, kg, calories):
            G = extract_gram(build_nutritional_values(kg, calories))
            E, F, P = G['Carbohydrates Grams'], G['Fat Grams'], G['Protein Grams']
            day_data = days_data[day]
            day_data = day_data[day_data.calories != 0]
            food, c, e, f, p = day_data.name.tolist(), day_data.calories.tolist(), day_data.carbohydrate.tolist(), \
                               day_data.total_fat.tolist(), day_data.protein.tolist()
            x = pulp.LpVariable.dicts("x", indices=food, lowBound=0, upBound=1.5, cat='Continuous', indexStart=[])
            prob = pulp.LpProblem("Diet", LpMinimize)
            prob += pulp.lpSum([x[food[i]] * c[i] for i in range(len(food))])
            prob += pulp.lpSum([x[food[i]] * e[i] for i in range(len(x))]) >= E
            prob += pulp.lpSum([x[food[i]] * f[i] for i in range(len(x))]) >= F
            prob += pulp.lpSum([x[food[i]] * p[i] for i in range(len(x))]) >= P
            try:
                prob.solve()
            except Exception as e:
                print("Error during optimization:", e)

            variables, values = [], []
            for v in prob.variables():
                variable, value = v.name, v.varValue
                variables.append(variable)
                values.append(value)
            values = np.array(values)
            if values is not None:
                values = values.round(2).astype(float)
            else:
                values = 0.0

            sol = pd.DataFrame(np.array([food, values]).T, columns=['Food', 'Quantity'])
            sol['Quantity'] = sol.Quantity.astype(float)
            sol = sol[sol['Quantity'] != 0.0]
            sol.Quantity = sol.Quantity * 100
            sol = sol.rename(columns={'Quantity': 'Quantity (g)'})
            return sol

        def total_model(kg, calories):
            result = {}
            for day in week_days:
                result[day] = model(day, kg, calories)
            return result

        # Get the diet plan based on user input
        diet = total_model(kg, calories)

        # Clear previous tab content
        self.tab_widget.clear()

        # Populate the QTabWidget with the diet plan
        for day, df in diet.items():
            table_widget = QTableWidget()
            table_widget.setColumnCount(3)  # 3 columns: Food, Quantity (g), Calorie
            table_widget.setHorizontalHeaderLabels(['Food', 'Quantity (g)', 'Calorie (kcal)'])

            # Filtering: Exclude quantities less than 0.05
            df = df[df['Quantity (g)'] >= 0.05]

            table_widget.setRowCount(len(df))
            x = 0
            # Iterate through each row of the filtered dataframe and populate the table
            for index, row in df.iterrows():
                food_item, quantity_item, calorie_item = (
                    QTableWidgetItem(row['Food']),
                    QTableWidgetItem(f"{row['Quantity (g)']:.1f}"),  # 2 decimal places
                    QTableWidgetItem(f"{row['Quantity (g)'] * data.loc[data['name'] == row['Food'], 'calories'].values[0] / 100.0:.1f}")
                )

                # Set the items in the table
                table_widget.setItem(x, 0, food_item)
                table_widget.setItem(x, 1, quantity_item)
                table_widget.setItem(x, 2, calorie_item)
                x += 1

            # Add a tab for each day
            table_widget.setColumnWidth(0, 800)
            self.tab_widget.addTab(table_widget, day)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    nutrition_app = NutritionApp()
    nutrition_app.show()
    sys.exit(app.exec_())
