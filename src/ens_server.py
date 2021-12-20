import os
import io
import json
import pandas as pd

from flask import Flask, render_template, url_for
from flask import redirect, Response
from flask import send_from_directory, flash
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from wtforms import StringField, SelectField, FloatField, IntegerField
from wtforms.validators import DataRequired, Optional
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from ensembles import RandomForestMSE, GradientBoostingMSE


app = Flask(__name__)
app.config['SECRET_KEY'] = 'hello'
app.url_map.strict_slashes = False

models = {}
datasets = {}


class Ensemble:

    """
    This class is mix of the 2.
    """

    __models = {
        'RF': RandomForestMSE,
        'GBM': GradientBoostingMSE,
    }

    __types = {
        'RF': 'Случайный лес',
        'GBM': 'Градиентный бустинг'
    }

    def __init__(self, name, ens_type, form):
        self.name = name
        hyparams = form.data
        d = [('Тип ансамбля', self.__types[ens_type])]
        d += [(form[param].label.text, hyparams[param]) for param in hyparams]
        self.description = pd.DataFrame(d, columns=['Параметр', 'Значение'])
        trees_parameters = hyparams.pop('trees_parameters')
        hyparams = {**hyparams, **trees_parameters}
        self.model = self.__models[ens_type](**hyparams)
        self.train_loss = None
        self.val_loss = None
        self.fitted_on = None
        self.target_name = None

    def fit(self, data_train, data_val=None):
        """
        Fitting model
        """
        if self.fitted_on is None:
            self.description = pd.concat([self.description, pd.DataFrame(
                [["Обучен на выборке", data_train.name]], columns=["Параметр", "Значение"])], axis=0)
        self.fitted_on = data_train.name
        X_train = data_train.features()
        y_train = data_train.target
        self.target_name = data_train.target_name
        if data_val is not None:
            self.train_loss, self.val_loss, _ = self.model.fit(
                X_train, y_train, data_val.features(),
                data_val.target)
        else:
            self.train_loss, _ = self.model.fit(X_train, y_train)

    @property
    def is_fitted(self):
        """
        Is fitted?
        """
        return self.train_loss is not None

    def predict(self, data_test):
        """
        Predicting
        """
        y_pred = self.model.predict(data_test.features())
        return pd.DataFrame(
            y_pred,
            index=data_test.data.index,
            columns=[self.target_name]
        )

    def plot(self):
        """
        Plotting graphics
        """
        plt.rc('font', family='serif')
        plt.rc('axes', axisbelow=True, grid=True)
        plt.rc('grid', c='grey', ls=':')
        plt.rc('mathtext', fontset='dejavuserif')
        plt.rc('savefig', facecolor='white')
        fig, ax = plt.subplots(figsize=(6, 3), dpi=500)
        ax.set_title('Значение ошибки во время обучения')
        lim = self.model.n_estimators
        ax.plot(np.arange(1, lim + 1), self.train_loss,
                label='On train', c='b')
        if self.val_loss is not None:
            ax.plot(np.arange(1, lim + 1), self.val_loss,
                    label='On validation', c='m')
        ax.set_xlabel('Количество обученных деревьев')
        ax.set_ylabel('RMSE')
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
        return fig


class Dataset:
    """
    This class shows dataset
    """
    def __init__(self, name, data, target_name):
        self.name = name
        self.data = data
        self.target_name = target_name
        self.has_target = target_name != ''

    @property
    def features(self):
        """
        Features
        """
        return self.data.drop(columns=self.target_name).to_numpy

    @property
    def target(self):
        """
        Computing target
        """
        return self.data[self.target_name].to_numpy()


model_types = [
    ('RF', 'Случайный лес'),
    ('GBM', 'Градиентный бустинг'),
]


def json_field_filter(str_):
    """
        Json args processing
    """
    try:
        dict_ = json.loads(str_)
    except TypeError:
        dict_ = {}
    return dict_


class NewEnsembleForm(FlaskForm):
    name = StringField('Название модели', validators=[DataRequired()])
    model_type = SelectField('Тип ансамбля', choices=model_types)
    n_estimators = IntegerField('Количество деревьев', default=100)
    learning_rate = FloatField('Темп обучения (только для бустинга)', default=0.1)
    max_depth = IntegerField('Максимальная глубина', validators=[Optional()])
    feature_subsample_size = IntegerField(
        'Размерность подвыборки признаков для одного дерева',
        validators=[Optional()],
    )
    trees_parameters = StringField(
        'Дополнительные параметры для дерева (вводятся в кавычках, через : и пробел)',
        validators=[Optional()],
        filters=[json_field_filter]
    )


class UploadForm(FlaskForm):
    name = StringField('Введите имя датасета', validators=[DataRequired()])
    features_file = FileField('Прикрепите файл с данными (csv)', validators=[
        FileRequired(),
        FileAllowed(['csv'], 'CSV only!')
    ])
    target_name = StringField('Введите имя целевой переменной', validators=[
        DataRequired(),
    ])


class LearnForm(FlaskForm):
    train_data = SelectField('Обучающая выборка', validators=[DataRequired()])
    val_data = SelectField('Валидационная выборка')


class TestForm(FlaskForm):
    test_data = SelectField('Тестовая выборка', validators=[DataRequired()])


@app.route('/index')
@app.route('/')
def get_index():
    """
    Main menu handler
    """
    return render_template('index.html')


@app.route('/models/', methods=['GET', 'POST'])
def get_models():
    """
    Model processor handler
    """
    form = NewEnsembleForm(meta={'csrf': False})
    if form.validate_on_submit():
        model_type = form.model_type.data
        name = form.name.data
        if model_type == 'RF':
            del form.learning_rate
        del form.model_type
        del form.name
        models[name] = Ensemble(name, model_type, form)
        return redirect(url_for('get_models'))
    return render_template('models.html', form=form, models=models)


@app.route('/data/', methods=['GET', 'POST'])
def get_data():
    """
    Data downloader handler
    """
    form = UploadForm()
    if form.validate_on_submit():
        data = pd.read_csv(form.features_file.data, index_col=0,
                           float_precision='round_trip')
        target_name = form.target_name.data
        if target_name not in data.columns:
            flash("Target not in columns", "error")
            return render_template('datasets.html', form=form, datasets=datasets)
        datasets[form.name.data] = Dataset(form.name.data, data, target_name)
        return redirect(url_for('get_data'))
    return render_template('datasets.html', form=form, datasets=datasets)


@app.route('/models/<name>/plot.png')
def plot_png(name):
    """
    Plotting a graph
    """
    fig = models[name].plot()
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/models/<name>/work/', methods=['GET', 'POST'])
def model_page(name):
    """
    Model fitting page
    """
    learn_form = LearnForm(meta={'csrf': False})
    test_form = TestForm(meta={'csrf': False})
    learn_form.train_data.choices = [d for d in datasets
                                     if datasets[d].has_target]
    learn_form.val_data.choices = ['-'] + learn_form.train_data.choices
    test_form.test_data.choices = list(datasets.keys())
    if learn_form.validate_on_submit():
        data_train = datasets[learn_form.train_data.data]
        data_val = datasets.get(learn_form.val_data.data)
        try:
            models[name].fit(data_train, data_val)
        except ValueError:
            data_train_trans = None
            data_val_trans = None
            flash("Data cannot transform into numpy, using OrdinalEncoder to solve the issue", "error")
            trf = ColumnTransformer([
              ('encoding', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), data_train.data.dtypes[
                  data_train.data.dtypes == "O"].index)
              ], remainder='passthrough')
            trf = trf.fit(data_train.data)
            data_train_trans = trf.transform(data_train.data)
            if data_val is not None:
                if (data_val.target_name != data_train.target_name) | (not
                        np.array_equal(data_val.data.columns, data_train.data.columns)):
                    flash("Data from different tables", "error")
                    return redirect(url_for('model_page', name=name))
                data_val_trans = trf.transform(data_val.data)
                models[name].fit(Dataset(data_train.name, pd.DataFrame(
                    data_train_trans, columns=data_train.data.columns, index=data_train.data.index
                        ), data_train.target_name), Dataset(data_val.name, pd.DataFrame(
                            data_val_trans, columns=data_val.data.columns, index=data_val.data.index
                                ), data_val.target_name))
            models[name].fit(Dataset(data_train.name, pd.DataFrame(
                    data_train_trans, columns=data_train.data.columns, index=data_train.data.index
                        ), data_train.target_name), None)
        return redirect(url_for('model_page', name=name))
    if test_form.validate_on_submit():
        data_train = datasets[models[name].fitted_on]
        data_test = datasets[test_form.test_data.data]
        try:
            preds = models[name].predict(data_test)
        except ValueError:
            trf = ColumnTransformer([
              ('encoding', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                  data_train.data.dtypes[data_train.data.dtypes == "O"].index)
              ], remainder='passthrough')
            trf = trf.fit(data_train.data)
            if len(data_test.data.columns) != len(data_train.data.columns):
                flash("Data from different tables", "error")
                return redirect(url_for('model_page', name=name))
            if (data_test.target_name != data_train.target_name) | (not
                    np.array_equal(data_test.data.columns, data_train.data.columns)):
                flash("Data from different tables", "error")
                return redirect(url_for('model_page', name=name))
            flash("Data cannot transform into numpy, using OrdinalEncoder to solve the issue", "error")
            data_test_trans = trf.transform(data_test.data)
            preds = models[name].predict(Dataset(data_test.name, pd.DataFrame(
                data_test_trans, columns=data_test.data.columns, index=data_test.data.index), data_train.target_name))
        fname = data_test.name + '_pred.csv'
        path = os.path.join(os.getcwd(), 'tmp/')
        if not os.path.exists(path):
            os.mkdir(path)
        preds.to_csv(os.path.join(path, fname))
        return send_from_directory(path, fname, as_attachment=True)
    template_kwargs = {
        'model': models[name],
        'learn_form': learn_form,
        'test_form': test_form,
        "datasets": datasets
    }
    return render_template('model_page.html', **template_kwargs)
