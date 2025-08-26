# Knapsack Problem Solver with Google OR-Tools

## Executive Summary

This project explores the relationship between Formula One driving style and performance from the last 50 years. This is based on K Means Clustering of the driver’s performance and grouped drivers into one of six clusters:

- **Elite Drivers**
- **Back-Markers**
- **Top Contenders**
- **Aggressive Risk Prone Drivers**
- **Consistent Midfielder**
- **Lower Midfield Fighters**
  
## Data Selection

The dataset **Formula 1 World Championship (1950 - 2024)** used for the project, sourced from Kaggle, included 14 CSV files however this project only utilised 5 of these. ["Formula 1 World Championships (1950-2024)"](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020). 

- ***drivers*** - 9 columns, 862 rows.
- ***qualifying*** - 9 columns, 10,494 rows.
- ***races*** - 18 columns, 1,125 rows.
- ***results*** - 18 columns, 26,759 rows.
- ***status*** - 2 columns, 139 rows.

### Loading and Initial Exploration

The five CSV files were imported into the Pandas DataFrame using the `pd.read_csv()` function.

![Screenshot: Upload Info](screenshots/01_initial_exploration/pandas_read_csv.png)

The initial stages of the analysis were to explore the quality of the dataset and merge the DataFrames together into one large DataFrame which contained all of the information from each CSV.

```python
# Access the dataframes from the named_dataframes dictionary
results_df = named_dataframes['results_df']
status_df = named_dataframes['status_df']
races_df = named_dataframes['races_df']
drivers_df = named_dataframes['drivers_df']
qualifying_df = named_dataframes['qualifying_df']

# Perform the first merge: results with status
merged_results_status = pd.merge(results_df, status_df, on='statusId', how='left')

# Perform the second merge: the result of the first merge with drivers
merged_results_driver = pd.merge(merged_results_status, drivers_df, on='driverId', how='left')

# Perform the third merge: the result of the second merge with races
merged_results_races = pd.merge(merged_results_driver, races_df, on='raceId', how='left')

# Perform the fourth merge: the result of the third merge with qualifying
final_merged_df = pd.merge(merged_results_races, qualifying_df, on=['raceId', 'driverId'], how='left')

print("Merged dataframes:")
display(final_merged_df.head())

### Data Selection and Renaming

To focus on the relevant columns for create groups of Formula 1 Drivers, the following columns were selected:

`bin_dataset = df[['Uniq Id','Shipping Weight','Selling Price']]`

These columns were then renamed as 'id', 'weight' and 'price' for clarity and consistency.
`bin_dataset = bin_dataset.rename(columns={'Uniq Id':'id','Shipping Weight':'weight','Selling Price':'price'})`

![Screenshot: Renamed Columns](screenshots/01_initial_exploration/screenshot2.png)

### Handling Missing Values

Rows containing missing values in the 'weight' and 'price' columns were dropped from the dataset to ensure data integrity and accuracy in subsequent calculations.

![Screenshot: Rows with Missing Values](screenshots/01_initial_exploration/screenshot3.png)

`bin_dataset.dropna(subset=['weight', 'price'],inplace=True)`

![Screenshot: Dropping Rows with Missing Values](screenshots/01_initial_exploration/screenshot4.png)

### Standardizing Weight Units

To standardize the weight units across the dataset, we extracted and converted the weight values to kilograms. 
This involved handling different units such as 'pounds' and 'ounces' by applying a conversion function.

![Screenshot: Weight Conversion](screenshots/01_initial_exploration/screenshot5.png)

There was one row which contained '.' instead of a numeric value, this row was dropped 

![Screenshot: Weight Conversion Errors](screenshots/01_initial_exploration/screenshot6.png)

### Handling Price Values

![Screenshot: Price Conversion](screenshots/01_initial_exploration/screenshot7.png)

We also cleaned and standardized the price values, removing non-numeric characters, currency symbols and errors from the scraped data.

![Screenshot: Price Conversion Errors](screenshots/01_initial_exploration/screenshot8.png)

### Column Removal

![Screenshot: Removing Columns](screenshots/01_initial_exploration/screenshot9.png)

The original 'price' and 'weight' columns were removed from the dataset, as they were replaced with the standardized 'price' and 'weight_kg' columns.

### Final Dataset

![Screenshot: Final Dataset](screenshots/01_initial_exploration/screenshot10.png)

The resulting cleaned and standardized dataset was saved for future use, and it contains 'id', 'weight_kg', and 'price' columns.

## Solving the Knapsack Problem

In this section, we walk through the steps of solving the Knapsack Problem using Google OR-Tools with the cleaned and standardized dataset. The Knapsack Problem involves selecting items to maximize value while staying within capacity constraints, which is a valuable optimization technique for retail organizations.

### Loading the Cleaned Dataset

We begin by loading the previously cleaned and standardized dataset, which includes 'id', 'weight_kg' and 'price' columns. 
This dataset is now ready for solving the Knapsack Problem.

![Screenshot: Loading the Cleaned Dataset](screenshots/02_knapsack_problem/screenshot1.png)

### Random Sampling for Reproducibility

For reproducibility, we set a random seed and shuffle the dataset. We then select a random subset of items to solve the Knapsack Problem.

```python
random_seed = 42
random.seed(random_seed)

shuffled_df = df.sample(frac=1, random_state=random_seed)

num_samples = 100
random_subset = shuffled_df.head(num_samples)
```
![Screenshot: Sampled Subset](screenshots/02_knapsack_problem/screenshot2.png)

### Knapsack Problem Solver

We initialize the Knapsack Solver with the desired solver type and set the capacity of the knapsack (maximum weight allowed). 
This solver type is suitable for multidimensional problems.

```python
solver = knapsack_solver.KnapsackSolver(
    knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
    "KnapsackExample",
)

# Capacity of the knapsack (maximum weight allowed)
capacity = 25000
```

### Preparing Data for the Solver

We prepare the data for the solver by converting weights and values to integers as required by OR-Tools. 
The number of items in the subset is also determined.

```python
weights = random_subset['weight_kg'].tolist()
values = random_subset['price'].tolist()

# Convert weights and values to integers (required by OR-Tools)
weights = [int(w * 1000) for w in weights]  # Convert to grams (integer)
values = [int(v * 100) for v in values]  # Convert to currency (integer)

# Number of items
num_items = len(random_subset)
```

### Solver Initialization and Solution

We initialize the solver with the prepared data and solve the Knapsack Problem to find the selected items (1 for selected, 0 for not selected).

```python
# Set the solver parameters
solver.init(values, [weights], [capacity])
solver.solve()

# Get the selected items (1 for selected, 0 for not selected)
selected_items = [solver.best_solution_contains(i) for i in range(num_items)]
```

### Results and Analysis

We present the output, and verify the results by summing the total price and weight. 
In this step, we encounter a discrepancy in the total weight, which prompts further investigation.

![Screenshot: Results and Analysis](screenshots/02_knapsack_problem/screenshot3.png)

![Screenshot: Results and Analysis Discrepancy](screenshots/02_knapsack_problem/screenshot4.png)

### Addressing Discrepancy

We address the discrepancy by converting the 'kg' weight column into grams. 
After performing the conversion and re-sampling, we solve the Knapsack Problem again.

```python
def convert_grams(weight_kg):
    weight_grams = math.ceil(weight_kg * 1000)

    try:
        return int(weight_grams)
    except ValueError:
        return None

df['weight_grams'] = df['weight_kg'].apply(convert_grams)
```

### Reanalysis of Results

We reanalyze the results with the converted weight in grams, including the total weight in kilograms and grams.
The discrepancy still exists, but the solver does get the total weight below the target of 25,000 grams or 25kg

![Screenshot: Reanalysis of Results](screenshots/02_knapsack_problem/screenshot5.png)

![Screenshot: Reanalysis of Results](screenshots/02_knapsack_problem/screenshot6.png)

## Recommendations for Future Iterations

1. **Multiple Knapsacks:** Future iterations of this project could expand from solving a single knapsack problem to solving multiple knapsacks simultaneously. This would be valuable for scenarios where there are multiple constraints or resources to consider.

2. **Complex Packing Problems:** Consider tackling more complex packing problems, such as 2D or 3D bin packing, which have practical applications in logistics, manufacturing, and resource allocation.

## Challenges Encountered

One challenge encountered during the project was dealing with floating-point arithmetic limitations, which can affect the precision of calculations when working with values like weights and prices.
While the discrepancy is small enough for this project, it's essential to be aware of such limitations in real-world applications.

## References

1. [Amazon Product Dataset 2020](https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020)
2. [Google OR-Tools Documentation](https://developers.google.com/optimization/pack/knapsack)
3. Medium Articles:
   - [Exploring the Bin Packing Problem](https://medium.com/swlh/exploring-the-bin-packing-problem-f54a93ebdbe5)
   - [The Knapsack Problem: A Guide to Solving One of the Classic Problems in Computer Science](https://levelup.gitconnected.com/the-knapsack-problem-a-guide-to-solving-one-of-the-classic-problems-in-computer-science-7b63b0851c89)
