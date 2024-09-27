from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import xarray as xr
import os
import dash
from dash import html, dcc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd 

path_repertory = "data"

mask = xr.open_dataset("data/era5_mask.zarr", engine='zarr')

metrics_list = ["Metrics On Forecast And Levels"]

def get_local_files(path):
    files = []
    for file_name in os.listdir(path):
        if not file_name.startswith('.'):
            files.append({'label': file_name, 'value': file_name})
    return files

def is_two_dimensions_of_var_in_list(data, list):
    count = sum([1 for var in list if var in data.dims])
    return count >= 2

def are_dimensions_distinct(data, list):
    standardized_dims = {
        'longitude': 'lon',
        'latitude': 'lat',
        'level': 'lev',
        'lon': 'longitude',
        'lat': 'latitude',
        'lev': 'level'
    }
    dims = [dim for dim in data.dims if dim in list or standardized_dims.get(dim) in list]
    if len(dims) < 2:
        return False
    return len(set(standardized_dims.get(dim, dim) for dim in dims)) == len(dims)

def get_filtered_plotable_variables(file, plotalbes_vars=True):
    dataset_path = f'{path_repertory}/{file}'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No such file or directory: '{dataset_path}'")
    ds = xr.open_zarr(dataset_path)
    variables = list(ds.data_vars.keys())
    potables_vars = ['longitude', 'latitude', 'level', 'lon', 'lat', 'lev']
    if plotalbes_vars:
        return [{'label': var, 'value': var} for var in variables 
                if is_two_dimensions_of_var_in_list(ds[var], potables_vars) 
                and are_dimensions_distinct(ds[var], potables_vars)]
    else:
        return [{'label': var, 'value': var} for var in variables 
                if not is_two_dimensions_of_var_in_list(ds[var], potables_vars) 
                or not are_dimensions_distinct(ds[var], potables_vars)]

def check_forecast_dimension(file):
    dataset_path = f'{path_repertory}/{file}'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No such file or directory: '{dataset_path}'")
    ds = xr.open_zarr(dataset_path)
    if 'forecast' in ds.dims:
        return len(ds['forecast'])
    return 0

def check_dimension(file, dim):
    dataset_path = f'{path_repertory}/{file}'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No such file or directory: '{dataset_path}'")
    ds = xr.open_zarr(dataset_path)
    return dim in ds.dims

def concatene_time(year, month, day, hour):
    if year is None or month is None or day is None or hour is None:
        return None
    month = f"{int(month):02d}"
    day = f"{int(day):02d}"
    hour = f"{int(hour):02d}"
    return np.datetime64(f"{year}-{month}-{day}T{hour}:00:00")


def get_start_and_end_time(start_year, start_month, start_day, start_hour, end_year, end_month, end_day, end_hour):
    start_time = concatene_time(start_year, start_month, start_day, start_hour)
    end_time = concatene_time(end_year, end_month, end_day, end_hour)
    return start_time, end_time

def is_period_in_dataset(dataset, start_time, end_time):
    if start_time is None or end_time is None:
        return False
    return start_time in dataset.time.values and end_time in dataset.time.values

def get_index_of_time(dataset, time):
    if time is None:
        return None
    index = np.where(dataset.time.values == time)
    return index[0][0] if len(index[0]) > 0 else None



def get_start_end_time(dataset: xr.Dataset):
    """Extrait le temps de début et de fin du jeu de données."""
    return dataset.time[0].values, dataset.time[-1].values


def get_year_start_end_time(dataset: xr.Dataset):
    """Extrait les années de début et de fin du jeu de données."""
    return dataset.time[0].dt.year.values, dataset.time[-1].dt.year.values

def all_year_between(dataset: xr.Dataset):
    """Renvoie toutes les années présentes dans le jeu de données."""
    start_year, end_year = get_year_start_end_time(dataset)
    return list(range(start_year, end_year + 1))


def know_step_hour_for_the_first_day(dataset: xr.Dataset):
    """Détermine le pas horaire pour le premier jour."""
    if len(dataset.time) < 2:
        return 1 
    return int((dataset.time[1] - dataset.time[0]).values / np.timedelta64(1, 'h'))



def all_the_hours_of_the_first_day(dataset: xr.Dataset):
    """Renvoie toutes les heures disponibles pour le premier jour."""
    step_hour = know_step_hour_for_the_first_day(dataset)
    return list(range(0, 24, step_hour))


def get_list_indexes_of_time(dataset, start_time, end_time):
    start_index = get_index_of_time(dataset, start_time)
    end_index = get_index_of_time(dataset, end_time)
    if start_index is not None and end_index is not None:
        return np.arange(start_index, end_index + 1)
    elif start_index is not None:
        return np.array([start_index])
    elif end_index is not None:
        return np.array([end_index])
    return np.array([])

def get_time_period(file):
    dataset_path = f'{path_repertory}/{file}'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No such file or directory: '{dataset_path}'")
    ds = xr.open_zarr(dataset_path)
    start_time = format_time(ds.time.values[0])
    end_time = format_time(ds.time.values[-1])
    return start_time, end_time

def format_time(np_datetime64):
    pd_timestamp = pd.to_datetime(np_datetime64)
    return pd_timestamp.strftime('%Y/%m/%d %H:%M')

def format_title(title_dict):
    lines = []
    if 'main_title' in title_dict and title_dict['main_title']:
        lines.append(title_dict['main_title'])
    subtitle_parts = []
    if 'dataset' in title_dict and title_dict['dataset']:
        subtitle_parts.append(f"Dataset: {title_dict['dataset']}")
    if 'ground_truth' in title_dict and title_dict['ground_truth']:
        subtitle_parts.append(f"Ground Truth: {title_dict['ground_truth']}")
    if 'variable' in title_dict and title_dict['variable']:
        subtitle_parts.append(f"Variable: {title_dict['variable']}")
    if 'level' in title_dict and title_dict['level'] is not None:
        subtitle_parts.append(f"Level: {title_dict['level']}")
    if 'forecast' in title_dict and title_dict['forecast'] is not None:
        subtitle_parts.append(f"Forecast: {title_dict['forecast']}")
    if subtitle_parts:
        lines.append(" | ".join(subtitle_parts))
    if 'time' in title_dict and title_dict['time']:
        lines.append(f"Time: {title_dict['time']}")
    return "\n".join(lines)


def plot_and_save_image(dataset, variable, time_index, x=None, y=None, rx=None, ry=None, level=None, forecast=None, mask=None, dataset_name=None):
    plt.figure(figsize=(8, 7))
    try:
        lat_dim = len(dataset.latitude)
        lon_dim = len(dataset.longitude)

        # Normalisation des coordonnées x et y
        if x is not None:
            x = x % lon_dim
        if y is not None:
            y = y % lat_dim

        # Sélection des données en fonction des dimensions disponibles
        if level is not None and forecast is not None:
            data = dataset[variable].isel(time=time_index, level=level, forecast=forecast)
        elif level is not None:
            data = dataset[variable].isel(time=time_index, level=level)
        elif forecast is not None:
            data = dataset[variable].isel(time=time_index, forecast=forecast)
        else:
            data = dataset[variable].isel(time=time_index)

        # Formatage du temps
        time_value = format_time(dataset.time.values[time_index])

        # Construction du titre avec les informations requises
        title_dict = {
            'dataset': dataset_name,
            'variable': variable,
            'time': time_value
        }
        title = format_title(title_dict)
        plt.title(title)

        # Reste du code pour l'affichage de l'image
        # Cas 1 : Pas de x et y, affichage de l'image complète
        if x is None and y is None:
            if data.ndim == 2:
                plt.imshow(np.rot90(data), cmap='jet')
            elif data.ndim > 2:
                data_2d = data.isel(**{dim: 0 for dim in data.dims if dim not in ['latitude', 'longitude', 'lat', 'lon']})
                plt.imshow(np.rot90(data_2d), cmap='jet')
            else:
                raise ValueError("Les données doivent être au moins 2D pour être tracées")
            plt.grid(True)

        # Cas 2 : x et y présents, affichage avec un scatter
        elif x is not None and y is not None and (rx is None or ry is None):
            data_rotated = np.rot90(data)
            plt.imshow(data_rotated, cmap='jet')
            plt.scatter(x, y, color='black')
            plt.text(x, y + 2, f'({x}, {y})', color='black', fontsize=12, ha='center')
            plt.grid(True)

        # Cas 3 : Zoom avec rx et ry
        elif x is not None and y is not None and rx is not None and ry is not None:
            data_rotated = np.rot90(data)
            x_min = max(0, x - rx)
            x_max = min(data_rotated.shape[1], x + rx)
            y_min = max(0, y - ry)
            y_max = min(data_rotated.shape[0], y + ry)
            zoomed_region = data_rotated[y_min:y_max, x_min:x_max]
            plt.imshow(zoomed_region, cmap='jet', extent=[x_min, x_max, y_min, y_max])
            plt.scatter(x, y, color='black')
            plt.text(x, y + 2, f'({x}, {y})', color='black', fontsize=12, ha='center')
            plt.grid(True)

        # Ajout du masque si disponible
        if mask is not None:
            mask_data = mask["land_sea_mask"].data
            mask_rotated = np.rot90(mask_data)
            if x is not None and y is not None and rx is not None and ry is not None:
                mask_zoomed = mask_rotated[y_min:y_max, x_min:x_max]
                plt.imshow(mask_zoomed, cmap='binary', alpha=0.5, extent=[x_min, x_max, y_min, y_max])
            else:
                plt.imshow(mask_rotated, cmap='binary', alpha=0.5)
            plt.grid(True)

        # Sauvegarde de l'image en mémoire
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_str = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close()

        return f"data:image/png;base64,{image_str}"

    except IndexError as e:
        raise ValueError(f"Erreur d'indexation : {e}")


def list_images_plot(dataset, variable, start_time, end_time, level=None, forecast=None, mask=None, x=None, y=None, rx=None, ry=None, dataset_name=None):
    time_indexes = get_list_indexes_of_time(dataset, start_time, end_time)
    images = []
    for time_index in time_indexes:
        images.append(plot_and_save_image(dataset, variable, time_index, x, y, rx, ry, level, forecast, mask, dataset_name))
    return images



def check_conditions_for_plot_forecast_statistics(data):
    required_dims = {'level', 'latitude', 'longitude', 'time', 'forecast'}
    data_dims = set(data.dims)
    
    if not required_dims.issubset(data_dims):
        return False

    if not np.issubdtype(data.dtype, np.number):
        return False
    
    return True


def plot_forecast_statistics(ax, forecast_data, ground_truth_data, time_index, lat, lon, rx, ry, num_forecasts, dataset_name=None, ground_truth_name=None):
    def valeur(dataset, time_index, level, lat_slice, lon_slice, forecast=None):
        """Extract mean values over a specified region."""
        lat_slice = [s % lat_dim for s in lat_slice]
        lon_slice = [s % lon_dim for s in lon_slice]

        region = dataset.isel(time=time_index, level=level).sel(
            latitude=slice(min(lat_slice), max(lat_slice)),
            longitude=slice(min(lon_slice), max(lon_slice))
        )
        if forecast is not None:
            region = region.isel(forecast=forecast)
        return region.mean(dim=('latitude', 'longitude')).values

    list_level = forecast_data.level.values

    # Get dimensions
    lat_dim = len(forecast_data.latitude)
    lon_dim = len(forecast_data.longitude)

    # Normalize coordinates
    lat = lat % lat_dim
    lon = lon % lon_dim

    # Define slices
    lat_slice = [(lat - ry) % lat_dim, (lat + ry) % lat_dim]
    lon_slice = [(lon - rx) % lon_dim, (lon + rx) % lon_dim]

    Y_q05 = []
    Y_q25 = []
    Y_q50 = []
    Y_q75 = []
    Y_q95 = []
    Y_ground_truth = []

    for level in range(len(list_level)):
        forecast_values = [
            valeur(forecast_data, time_index, level, lat_slice, lon_slice, forecast) for forecast in range(num_forecasts)
        ]
        forecast_values = np.array(forecast_values)

        q05 = np.quantile(forecast_values, 0.05)
        q25 = np.quantile(forecast_values, 0.25)
        q50 = np.quantile(forecast_values, 0.5)
        q75 = np.quantile(forecast_values, 0.75)
        q95 = np.quantile(forecast_values, 0.95)

        Y_q05.append(q05)
        Y_q25.append(q25)
        Y_q50.append(q50)
        Y_q75.append(q75)
        Y_q95.append(q95)

        if ground_truth_data is not None:
            try:
                ground_truth_value = valeur(ground_truth_data, time_index, level, lat_slice, lon_slice)
                Y_ground_truth.append(ground_truth_value)
            except KeyError:
                Y_ground_truth.append(None)

    X = list_level
    Y_q05 = np.array(Y_q05)
    Y_q25 = np.array(Y_q25)
    Y_q50 = np.array(Y_q50)
    Y_q75 = np.array(Y_q75)
    Y_q95 = np.array(Y_q95)
    Y_ground_truth = np.array([gt for gt in Y_ground_truth if gt is not None])

    # Plot percentiles
    ax.fill_between(X, Y_q05, Y_q25, color='yellow', alpha=0.4, label='Q05 - Q25')
    ax.fill_between(X, Y_q25, Y_q50, color='orange', alpha=0.4, label='Q25 - Q50')
    ax.fill_between(X, Y_q50, Y_q75, color='red', alpha=0.4, label='Q50 - Q75')
    ax.fill_between(X, Y_q75, Y_q95, color='purple', alpha=0.4, label='Q75 - Q95')

    ax.plot(X, Y_q05, color='yellow')
    ax.plot(X, Y_q25, color='orange')
    ax.plot(X, Y_q50, color='red')
    ax.plot(X, Y_q75, color='purple')
    ax.plot(X, Y_q95, color='blue', label='Q95')

    if len(Y_ground_truth) > 0:
        ax.plot(X, Y_ground_truth, label='Ground Truth', color='black', linestyle='-', linewidth=2)

    ax.set_xlabel('Pressure Level')
    ax.set_ylabel('Variable Value')
    ax.legend()
    ax.grid(True)

def register_callbacks(app):
    # Callback pour mettre à jour les fichiers dans les dropdowns
    @app.callback(
        [Output('file-dropdown1', 'options'),
         Output('file-dropdown2', 'options')],
        Input('generate-button1', 'n_clicks')
    )
    def update_file_dropdowns(n_clicks):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate
        ref_files = get_local_files(path_repertory)
        pred_files = get_local_files(path_repertory)
        return ref_files, pred_files

    # Callback pour mettre à jour les variables disponibles pour le fichier 1
    @app.callback(
        [Output('variable-dropdown1', 'options'),
         Output('forecast-number-container1', 'style'),
         Output('forecast-number-input1', 'max'),
         Output('time-period1', 'children')],
        Input('file-dropdown1', 'value')
    )
    def update_variable_dropdown1(selected_file):
        if selected_file is None:
            return [], {'display': 'none'}, None, ''
        try:
            variables = get_filtered_plotable_variables(selected_file)
            forecast_length = check_forecast_dimension(selected_file)
            start_time, end_time = get_time_period(selected_file)
            time_period_text = f"Time Period: {start_time} to {end_time}"
            if forecast_length > 0:
                forecast_style = {'display': 'block', 'marginBottom': '20px'}
            else:
                forecast_style = {'display': 'none'}
            return variables, forecast_style, forecast_length, time_period_text
        except FileNotFoundError as e:
            return [], {'display': 'none'}, None, str(e)

    # Callback pour mettre à jour les variables disponibles pour le fichier 2
    @app.callback(
        [Output('variable-dropdown2', 'options'),
         Output('forecast-number-container2', 'style'),
         Output('forecast-number-input2', 'max'),
         Output('time-period2', 'children')],
        Input('file-dropdown2', 'value')
    )
    def update_variable_dropdown2(selected_file):
        if selected_file is None:
            return [], {'display': 'none'}, None, ''
        try:
            variables = get_filtered_plotable_variables(selected_file)
            forecast_length = check_forecast_dimension(selected_file)
            start_time, end_time = get_time_period(selected_file)
            time_period_text = f"Time Period: {start_time} to {end_time}"
            if forecast_length > 0:
                forecast_style = {'display': 'block', 'marginBottom': '20px'}
            else:
                forecast_style = {'display': 'none'}
            return variables, forecast_style, forecast_length, time_period_text
        except FileNotFoundError as e:
            return [], {'display': 'none'}, None, str(e)


    # Callback pour mettre à jour le dropdown des niveaux pour le graphique 1
    @app.callback(
        [Output('range-z1-dropdown', 'options'),
         Output('range-z1-dropdown', 'value')],
        Input('variable-dropdown1', 'value'),
        State('file-dropdown1', 'value')
    )
    def update_level_dropdown1(variable, selected_file):
        if not variable or not selected_file:
            return [], None

        dataset_path = f'{path_repertory}/{selected_file}'
        ds = xr.open_zarr(dataset_path)
        levels = ds[variable].level.values if 'level' in ds[variable].dims else []

        options = [{'label': str(level), 'value': level} for level in levels]

        return options, None

    # Callback pour mettre à jour le dropdown des niveaux pour le graphique 2
    @app.callback(
        [Output('range-z2-dropdown', 'options'),
         Output('range-z2-dropdown', 'value')],
        Input('variable-dropdown2', 'value'),
        State('file-dropdown2', 'value')
    )
    def update_level_dropdown2(variable, selected_file):
        if not variable or not selected_file:
            return [], None

        dataset_path = f'{path_repertory}/{selected_file}'
        ds = xr.open_zarr(dataset_path)
        levels = ds[variable].level.values if 'level' in ds[variable].dims else []

        options = [{'label': str(level), 'value': level} for level in levels]

        return options, None

    # Callback pour générer et mettre à jour les images des graphiques 1 et 2
    @app.callback(
        [Output('stored-images1', 'data'),
         Output('stored-images2', 'data'),
         Output('graph1-plot', 'src'),
         Output('graph2-plot', 'src'),
         Output('time-period11', 'children'),
         Output('interval', 'disabled')],
        [Input('generate-button1', 'n_clicks'),
         Input('prev-button1', 'n_clicks'),
         Input('next-button1', 'n_clicks'),
         Input('play-button', 'n_clicks'),
         Input('interval', 'n_intervals')],
        [State('file-dropdown1', 'value'),
         State('variable-dropdown1', 'value'),
         State('start-year-dropdown', 'value'),
         State('start-month-dropdown', 'value'),
         State('start-day-dropdown', 'value'),
         State('start-hour-dropdown', 'value'),
         State('end-year-dropdown', 'value'),
         State('end-month-dropdown', 'value'),
         State('end-day-dropdown', 'value'),
         State('end-hour-dropdown', 'value'),
         State('forecast-number-input1', 'value'),
         State('range-z1-dropdown', 'value'),
         State('file-dropdown2', 'value'),
         State('variable-dropdown2', 'value'),
         State('forecast-number-input2', 'value'),
         State('range-z2-dropdown', 'value'),
         State('range-x', 'value'),
         State('range-y', 'value'),
         State('range-rx', 'value'),
         State('range-ry', 'value'),
         State('stored-images1', 'data'),
         State('stored-images2', 'data'),
         State('graph1-plot', 'src'),
         State('graph2-plot', 'src'),
         State('interval', 'disabled')]
    )
    def generate_and_update_images(n_clicks_generate, n_clicks_prev, n_clicks_next, n_clicks_play, n_intervals,
                                   file1, variable1, start_year, start_month, start_day, start_hour,
                                   end_year, end_month, end_day, end_hour,
                                   forecast1, z1, file2, variable2, forecast2, z2, x, y, rx, ry,
                                   stored_images1, stored_images2, current_src1, current_src2, interval_disabled):
        ctx = dash.callback_context
        button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        # Initialiser les variables
        images1 = stored_images1 if stored_images1 else []
        images2 = stored_images2 if stored_images2 else []
        time_period_text = dash.no_update

        # Si le bouton "Generate" est cliqué, régénérer les images et réinitialiser l'affichage
        if button_id == 'generate-button1':
            # Vérifier que toutes les valeurs de date sont sélectionnées
            if None in [start_year, start_month, start_day, start_hour,
                        end_year, end_month, end_day, end_hour]:
                raise PreventUpdate

            # Convertir les valeurs de date en datetime64
            start_time = concatene_time(start_year, start_month, start_day, start_hour)
            end_time = concatene_time(end_year, end_month, end_day, end_hour)

            if not start_time or not end_time:
                raise PreventUpdate

            images1, images2 = [], []

            if file1 and variable1:
                dataset_name1 = file1
                try:
                    ds1 = xr.open_zarr(f'{path_repertory}/{file1}')
                except FileNotFoundError:
                    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Dataset not found", interval_disabled

                if not is_period_in_dataset(ds1, start_time, end_time):
                    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Specified time period is not in the dataset", interval_disabled

                level_index1 = list(ds1[variable1].level.values).index(z1) if z1 else None
                images1 = list_images_plot(ds1, variable1, start_time, end_time, level_index1, forecast1, mask, x, y, rx, ry, dataset_name1)

            if file2 and variable2:
                dataset_name2 = file2
                try:
                    ds2 = xr.open_zarr(f'{path_repertory}/{file2}')
                except FileNotFoundError:
                    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Dataset not found", interval_disabled

                if not is_period_in_dataset(ds2, start_time, end_time):
                    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Specified time period is not in the dataset", interval_disabled

                level_index2 = list(ds2[variable2].level.values).index(z2) if z2 else None
                images2 = list_images_plot(ds2, variable2, start_time, end_time, level_index2, forecast2, mask, x, y, rx, ry, dataset_name2)

            if not images1 and not images2:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "No images generated", interval_disabled

            time_period_text = f"Time Period: {format_time(start_time)} to {format_time(end_time)}"

            interval_disabled = True

            return images1, images2, images1[0] if images1 else dash.no_update, images2[0] if images2 else dash.no_update, time_period_text, interval_disabled


        elif button_id in ['prev-button1', 'next-button1', 'play-button', 'interval']:
            if not stored_images1 and not stored_images2:
                raise PreventUpdate

            current_index1 = stored_images1.index(current_src1) if stored_images1 and current_src1 in stored_images1 else 0
            current_index2 = stored_images2.index(current_src2) if stored_images2 and current_src2 in stored_images2 else 0

            if button_id == 'prev-button1':
                if stored_images1:
                    current_index1 = (current_index1 - 1) % len(stored_images1)
                if stored_images2:
                    current_index2 = (current_index2 - 1) % len(stored_images2)
            elif button_id in ['next-button1', 'interval']:
                if stored_images1:
                    current_index1 = (current_index1 + 1) % len(stored_images1)
                if stored_images2:
                    current_index2 = (current_index2 + 1) % len(stored_images2)
            elif button_id == 'play-button':
                interval_disabled = not interval_disabled  

            return (
                dash.no_update,
                dash.no_update,
                stored_images1[current_index1] if stored_images1 else dash.no_update,
                stored_images2[current_index2] if stored_images2 else dash.no_update,
                dash.no_update,
                interval_disabled
            )

        raise PreventUpdate

    # Callback pour mettre à jour les heures disponibles
    @app.callback(
        [Output('start-hour-dropdown', 'options'),
         Output('end-hour-dropdown', 'options')],
        Input('file-dropdown1', 'value')
    )
    def update_hour_dropdowns(selected_file):
        if selected_file is None:
            return [], []
        try:
            dataset_path = f'{path_repertory}/{selected_file}'
            ds = xr.open_zarr(dataset_path)
            hours = all_the_hours_of_the_first_day(ds)
            hour_options = [{'label': f"{h:02d}:00", 'value': h} for h in hours]
            return hour_options, hour_options
        except Exception as e:
            print(f"Error updating hour dropdowns: {e}")
            return [], []

    @app.callback(
        [Output('start-year-dropdown', 'options'),
         Output('end-year-dropdown', 'options')],
        Input('file-dropdown1', 'value')
    )
    def update_year_dropdowns(selected_file):
        if selected_file is None:
            return [], []
        try:
            dataset_path = f'{path_repertory}/{selected_file}'
            ds = xr.open_zarr(dataset_path)
            years = all_year_between(ds)
            year_options = [{'label': str(y), 'value': y} for y in years]
            return year_options, year_options
        except Exception as e:
            print(f"Error updating year dropdowns: {e}")
            return [], []

    @app.callback(
        Output('metrics-variables-options2', 'options'),
        Input('variable-dropdown2', 'value'),
        State('file-dropdown2', 'value')
    )
    def update_metrics_selection_dropdown(variable, selected_file):
        if variable is None or selected_file is None:
            return []

        try:
            dataset_path = f'{path_repertory}/{selected_file}'
            ds = xr.open_zarr(dataset_path)
            data_var = ds[variable]
            metrics = []

            if check_conditions_for_plot_forecast_statistics(data_var):
                metrics.append({'label': 'Forecast Statistics', 'value': 'forecast_statistics'})

            return metrics
        except FileNotFoundError:
            return []

    @app.callback(
        [Output('stored-images3', 'data'),
         Output('graph3-plot', 'src')],
        [Input('generate-metrics-button', 'n_clicks'),
         Input('prev-button2', 'n_clicks'),
         Input('next-button2', 'n_clicks')],
        [State('file-dropdown2', 'value'),
         State('file-dropdown1', 'value'),  # Ground Truth
         State('variable-dropdown2', 'value'),
         State('start-year-dropdown', 'value'),
         State('start-month-dropdown', 'value'),
         State('start-day-dropdown', 'value'),
         State('start-hour-dropdown', 'value'),
         State('end-year-dropdown', 'value'),
         State('end-month-dropdown', 'value'),
         State('end-day-dropdown', 'value'),
         State('end-hour-dropdown', 'value'),
         State('range-x', 'value'),
         State('range-y', 'value'),
         State('range-rx', 'value'),
         State('range-ry', 'value'),
         State('metrics-variables-options2', 'value'),  
         State('stored-images3', 'data'),
         State('graph3-plot', 'src')]
    )
    def manage_graph3_images(n_clicks_generate, n_clicks_prev, n_clicks_next,
                             selected_file, selected_ground_truth, variable,
                             start_year, start_month, start_day, start_hour,
                             end_year, end_month, end_day, end_hour,
                             x, y, rx, ry, selected_metric,
                             stored_images, current_src):
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        def safe_get_start_end_time(s_year, s_month, s_day, s_hour, e_year, e_month, e_day, e_hour):
            s_hour = s_hour if s_hour is not None else 0
            e_hour = e_hour if e_hour is not None else 0
            start_time = concatene_time(s_year, s_month, s_day, s_hour)
            end_time = concatene_time(e_year, e_month, e_day, e_hour)
            return start_time, end_time

        if triggered_id == 'generate-metrics-button':
            required_inputs = [selected_file, selected_ground_truth, variable, start_year, start_month, start_day,
                               end_year, end_month, end_day, x, y, selected_metric]

            if not all(required_inputs):
                raise dash.exceptions.PreventUpdate

            if selected_metric != 'forecast_statistics':
                raise dash.exceptions.PreventUpdate

            start_time, end_time = safe_get_start_end_time(
                start_year, start_month, start_day, start_hour,
                end_year, end_month, end_day, end_hour
            )

            try:
                ds_forecast = xr.open_zarr(f'{path_repertory}/{selected_file}')
                dataset_name = selected_file
                lat_dim = len(ds_forecast.latitude)
                lon_dim = len(ds_forecast.longitude)

                # Normaliser les coordonnées avec modulo
                x = x % lon_dim
                y = y % lat_dim

                ds_ground_truth = None
                ground_truth_name = None
                if variable in xr.open_zarr(f'{path_repertory}/{selected_ground_truth}').variables:
                    ds_ground_truth = xr.open_zarr(f'{path_repertory}/{selected_ground_truth}')
                    ground_truth_name = selected_ground_truth
            except FileNotFoundError:
                return dash.no_update, dash.no_update

            if not is_period_in_dataset(ds_forecast, start_time, end_time):
                return dash.no_update, dash.no_update

            time_indexes = get_list_indexes_of_time(ds_forecast, start_time, end_time)
            num_forecasts = len(ds_forecast['forecast'])

            images = []
            for time_index in time_indexes:
                buf = BytesIO()
                plt.figure(figsize=(16, 9))
                ax = plt.gca()

                plot_forecast_statistics(
                    ax,
                    ds_forecast[variable],
                    ds_ground_truth[variable] if ds_ground_truth is not None else None,
                    time_index,
                    y,
                    x,
                    rx,
                    ry,
                    num_forecasts,
                    dataset_name=dataset_name,
                    ground_truth_name=ground_truth_name
                )

                time_value = format_time(ds_forecast.time.values[time_index])

                title_dict = {
                    'main_title': 'Forecast Statistics',
                    'dataset': dataset_name,
                    'ground_truth': ground_truth_name,
                    'variable': variable,
                    'time': time_value
                }
                title = format_title(title_dict)
                ax.set_title(title)

                plt.savefig(buf, format="png", bbox_inches='tight')
                buf.seek(0)
                image_str = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()
                plt.close()
                images.append(f"data:image/png;base64,{image_str}")

            if not images:
                return dash.no_update, dash.no_update

            return images, images[0]

        elif triggered_id in ['prev-button2', 'next-button2']:
            if not stored_images:
                raise dash.exceptions.PreventUpdate

            current_index = stored_images.index(current_src) if current_src in stored_images else 0

            if triggered_id == 'prev-button2':
                current_index = (current_index - 1) % len(stored_images)
            elif triggered_id == 'next-button2':
                current_index = (current_index + 1) % len(stored_images)

            return stored_images, stored_images[current_index]

        raise PreventUpdate
