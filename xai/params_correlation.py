import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

methods = ['pearson', 'spearman', 'kendall']  # list of methods
xl = pd.ExcelFile("DataForCorrelCalc.xlsx")  # read XLS-file
df = xl.parse("texture")  # read sheet with data

for method in methods:
    corr = df.corr(method=method)  # correlation dataframe
    sns.heatmap(corr, annot=True, linewidth=0.5)  # heatmap by seaborn
    plt.title('Correlation method: ' + method)  # title of whole image
    plt.xticks(rotation=45)  # rotation of X-axis label
    plt.show()  # show image