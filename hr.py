import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from flask import Flask,  render_template, request
from io import BytesIO
import base64


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('form.html')

@app.route('/data', methods=['POST'])
def model():
    df = pd.read_csv('data-final.csv', sep='\t')
    X = df.copy()

    X.drop(X.columns[50:107], axis=1, inplace=True)
    X.drop(X.columns[51:], axis=1, inplace=True)

    X.isnull().values.any()
    X.isnull().values.sum()
    X.dropna(inplace=True)

    df2 = X.drop('country', axis=1)
    columns = list(df2.columns)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df2 = scaler.fit_transform(df2)
    df2 = pd.DataFrame(df2, columns=columns)

    df2_model = X.drop('country', axis=1)
    kk = MiniBatchKMeans(n_clusters=5, random_state=0, batch_size=2048, max_iter=100)
    kkk = kk.fit(df2_model)

    clusters_numbers = kkk.labels_
    df2_model['Clusters'] = clusters_numbers

    col_list = list(df2_model)
    ext = col_list[0:10]
    est = col_list[10:20]
    agr = col_list[20:30]
    csn = col_list[30:40]
    opn = col_list[40:50]

    data_sums = pd.DataFrame()
    data_sums['extroversion'] = df2_model[ext].sum(axis=1) / 10
    data_sums['neurotic'] = df2_model[est].sum(axis=1) / 10
    data_sums['agreeable'] = df2_model[agr].sum(axis=1) / 10
    data_sums['conscientious'] = df2_model[csn].sum(axis=1) / 10
    data_sums['open'] = df2_model[opn].sum(axis=1) / 10
    data_sums['clusters'] = clusters_numbers
    data_sums.groupby('clusters').mean()

    a = np.array([[request.form["EXT1"],
          request.form["EXT2"],
          request.form["EXT3"],
          request.form["EXT4"],
          request.form["EXT5"],
          request.form["EXT6"],
          request.form["EXT7"],
          request.form["EXT8"],
          request.form["EXT9"],
          request.form["EXT10"],
          request.form["EST1"],
          request.form["EST2"],
          request.form["EST3"],
          request.form["EST4"],
          request.form["EST5"],
          request.form["EST6"],
          request.form["EST7"],
          request.form["EST8"],
          request.form["EST9"],
          request.form["EST10"],
          request.form["AGR1"],
          request.form["AGR2"],
          request.form["AGR3"],
          request.form["AGR4"],
          request.form["AGR5"],
          request.form["AGR6"],
          request.form["AGR7"],
          request.form["AGR8"],
          request.form["AGR9"],
          request.form["AGR10"],
          request.form["CSN1"],
          request.form["CSN2"],
          request.form["CSN3"],
          request.form["CSN4"],
          request.form["CSN5"],
          request.form["CSN6"],
          request.form["CSN7"],
          request.form["CSN8"],
          request.form["CSN9"],
          request.form["CSN10"],
          request.form["OPN1"],
          request.form["OPN2"],
          request.form["OPN3"],
          request.form["OPN4"],
          request.form["OPN5"],
          request.form["OPN6"],
          request.form["OPN7"],
          request.form["OPN8"],
          request.form["OPN9"],
          request.form["OPN10"]]])

    b = a.astype(int)
    aa = pd.DataFrame(b)

    predictions = kkk.predict(aa)

    col_list = list(aa)
    ext = col_list[0:10]
    est = col_list[10:20]
    agr = col_list[20:30]
    csn = col_list[30:40]
    opn = col_list[40:50]

    my_sums = pd.DataFrame()
    my_sums['Extroversion'] = aa[ext].sum(axis=1) / 10
    my_sums['Neurotic'] = aa[est].sum(axis=1) / 10
    my_sums['Agreeable'] = aa[agr].sum(axis=1) / 10
    my_sums['Conscientious'] = aa[csn].sum(axis=1) / 10
    my_sums['Openness'] = aa[opn].sum(axis=1) / 10
    my_sums['cluster'] = predictions

    job_list = {'AI Engineer': my_sums['Extroversion'][0] >= 2.1 and my_sums['Agreeable'][0] >= 2.7 and
                               my_sums['Conscientious'][0] >= 3.1 and my_sums['Neurotic'][0] >= 2.6 and
                               my_sums['Openness'][0] >= 3.1,
                'Project Manager':  my_sums['Extroversion'][0] >= 3.4 and my_sums['Agreeable'][0] >= 3.1 and
                                    my_sums['Conscientious'][0] >= 3.2 and my_sums['Neurotic'][0] >= 2.3 and
                                    my_sums['Openness'][0] >= 2.3,
                'Web Developer': my_sums['Extroversion'][0] >= 2.1 and my_sums['Agreeable'][0] >= 2.0 and
                                 my_sums['Conscientious'][0] >= 3.2 and my_sums['Neurotic'][0] >= 2.5 and
                                 my_sums['Openness'][0] >= 3.0,
                'Sales Manager': my_sums['Extroversion'][0] >= 3.4 and my_sums['Agreeable'][0] >= 3.3 and
                                 my_sums['Conscientious'][0] >= 3.4 and my_sums['Neurotic'][0] >= 3.1 and
                                 my_sums['Openness'][0] >= 2.6,
                'Marketing': my_sums['Extroversion'][0] >= 3.5 and my_sums['Agreeable'][0] >= 3.5 and
                             my_sums['Conscientious'][0] >= 2.5 and my_sums['Neurotic'][0] >= 3.3 and
                             my_sums['Openness'][0] >= 2.8,
                'Supervisor': my_sums['Extroversion'][0] >= 3.1 and my_sums['Agreeable'][0] >= 3.5 and
                              my_sums['Conscientious'][0] >= 3.1 and my_sums['Neurotic'][0] >= 3.2 and
                              my_sums['Openness'][0] >= 3.0,
                'Receptionist': my_sums['Extroversion'][0] >= 2.8 and my_sums['Agreeable'][0] >= 3.2 and
                                my_sums['Conscientious'][0] >= 3.3 and my_sums['Neurotic'][0] >= 3.4 and
                                my_sums['Openness'][0] >= 2.5
                }

    accept = ()
    reject = ()

    for i in range(0, 1):
        if request.values['job'] == 'AI Engineer' and job_list['AI Engineer']:
            accept = ('Congrats! You got the job ')
        elif request.values['job'] == 'Project Manager' and job_list['Project Manager']:
            accept = ('Congrats! You got the job ')
        elif request.values['job'] == 'Web Developer' and job_list['Web Developer']:
            accept = ('Congrats! You got the job ')
        elif request.values['job'] == 'Sales Manager' and job_list['Sales Manager']:
            accept = ('Congrats! You got the job ')
        elif request.values['job'] == 'Marketing' and job_list['Marketing']:
            accept = ('Congrats! You got the job ')
        elif request.values['job'] == 'Supervisor' and job_list['Supervisor']:
            accept = ('Congrats! You got the job ')
        elif request.values['job'] == 'Receptionist' and job_list['Receptionist']:
            accept = ('Congrats! You got the job ')
        else:
            reject = ("Unfortunately, you didn't get the job ")
            break



    my_sum = my_sums.drop('cluster', axis=1)
    plt.bar(my_sum.columns, my_sum.iloc[0, :], color='green', alpha=0.2)
    plt.plot(my_sum.columns, my_sum.iloc[0, :], color='red')
    plt.title(my_sums.iloc[0, 5])
    plt.xticks(rotation=15)
    plt.ylim(0, 5);
    cluss = BytesIO()
    plt.savefig(cluss, format="png")

    cluss.seek(0)
    plot_url = base64.b64encode(cluss.getvalue()).decode('utf8')
    job_applied = request.values["job"]

    if accept is ('Congrats! You got the job '):
        return render_template('plot.html',plot_url=plot_url, key=predictions, key1=job_applied, key2=accept)
    elif reject is ("Unfortunately, you didn't get the job "):
            return render_template('plot1.html',plot_url=plot_url, key=predictions, key1=job_applied, key3=reject)
    else:
        return None




if __name__ == '__main__':
    app.run(debug=True)

