import gc
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import tqdm as notebook_tqdm
import numpy as np
import pandas as pd
import itertools
from sklearn.utils import class_weight


def reduce_mem_usage(df, int_cast=True, obj_to_category=True, subset=None):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    Перебирает все колонки датафрема и изменяет тип данных с целью уменьшить размер датафрейма
    :param df: датафрейм (pd.DataFrame)
    :param int_cast: флаг реобразования в int (bool)
    :param obj_to_category: флаг преобразования в dtype (bool)
    :param subset: подмножество столбцов для преобразований (list)
    :return: датафрейм с преобразованными типами столбцов (pd.DataFrame)
    """
    start_mem = df.memory_usage().sum() / 1024 ** 3;
    gc.collect()
    print('Используемая память до преобразования {:.3f} Gb'.format(start_mem))

    cols = subset if subset is not None else df.columns.tolist()

    for col in tqdm(cols):

        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()

            # test if column can be converted to an integer
            treat_as_int = str(col_type)[:3] == 'int'
            #if int_cast and not treat_as_int:
                #treat_as_int = check_if_integer(df[col])

            if treat_as_int:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif 'datetime' not in col_type.name and obj_to_category:
            df[col] = df[col].astype('category')
    gc.collect()
    end_mem = df.memory_usage().sum() / 1024 ** 3
    print('Используемая память после преобразования: {:.3f} Gb'.format(end_mem))
    print('Количество используемой памяти уменьшено на {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
 
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def compute_weights(y: pd.DataFrame) -> np.array:
    """
    Функция возвращает массив весов для передачи в параметр weight lgb.Dataset
    
    :y: pd.DataFrame
        массив меток для вычисления весов классов
        
    :return: np.array
        массив размерности входного, содержащий веса классов
    """

    class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                  classes=np.unique(y), y=y)
    print(f'Веса рассчитанные compute_weights: {class_weights}')
    weights = np.zeros_like(y).astype(np.float32)
    weights[y == 0] = class_weights[0]
    weights[y == 1] = class_weights[1]
    return weights