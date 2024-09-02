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

path_repertory = "fake_data"

mask = xr.open_dataset("fake_data/era5_mask.zarr", engine='zarr')

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
    month = f"0{month}" if len(str(month)) == 1 else str(month)
    day = f"0{day}" if len(str(day)) == 1 else str(day)
    hour = f"0{hour}" if len(str(hour)) == 1 else str(hour)
    return np.datetime64(f"{year}-{month}-{day}T{hour}:00:00.000000000")

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
    start_time = str(ds.time.values[0])
    end_time = str(ds.time.values[-1])
    return start_time, end_time

def plot_and_save_image(dataset, variable, time_index, x=None, y=None, rx=None, ry=None, level=None, forecast=None, mask=None):
    plt.figure(figsize=(8, 7))
    try:
        lat_dim = len(dataset.latitude)
        lon_dim = len(dataset.longitude)

        # Normaliser les coordonnées x et y pour les ramener dans les bornes valides
        if x is not None:
            x = x % lon_dim
        if y is not None:
            y = y % lat_dim

        if level is not None and forecast is not None:
            data = dataset[variable].isel(time=time_index, level=level, forecast=forecast)
        elif level is not None:
            data = dataset[variable].isel(time=time_index, level=level)
        elif forecast is not None:
            data = dataset[variable].isel(time=time_index, forecast=forecast)
        else:
            data = dataset[variable].isel(time=time_index)

        # Cas 1: Si x et y sont None, on affiche le graphique sans rien
        if x is None and y is None:
            if data.ndim == 2:
                plt.imshow(np.rot90(data), cmap='coolwarm')
            elif data.ndim > 2:
                data_2d = data.isel(**{dim: 0 for dim in data.dims if dim not in ['latitude', 'longitude', 'lat', 'lon']})
                plt.imshow(np.rot90(data_2d), cmap='coolwarm')
            else:
                raise ValueError("Data must be at least 2D for plotting")
            plt.grid(True)

        # Cas 2: Si x et y sont présents, mais pas rx et ry, afficher le graphique avec scatter
        elif x is not None and y is not None and (rx is None or ry is None):
            data_rotated = np.rot90(data)
            plt.imshow(data_rotated, cmap='coolwarm')
            plt.scatter(x, y, color='black')  # Placer un point à la position (x, y)
            plt.text(x, y + 2, f'({x}, {y})', color='black', fontsize=12, ha='center')  # Afficher les coordonnées au-dessus du point
            plt.grid(True)

        # Cas 3: Si x, y, rx, et ry sont présents, afficher le zoom avec scatter
        elif x is not None and y is not None and rx is not None and ry is not None:
            data_rotated = np.rot90(data)
            # Calcul des limites pour le zoom
            x_min = max(0, x - rx)
            x_max = min(data_rotated.shape[1], x + rx)
            y_min = max(0, y - ry)
            y_max = min(data_rotated.shape[0], y + ry)
            
            # Extraire la région zoomée
            zoomed_region = data_rotated[y_min:y_max, x_min:x_max]
            plt.imshow(zoomed_region, cmap='coolwarm', extent=[x_min, x_max, y_min, y_max])
            plt.scatter(x, y, color='black')  # Placer un point à la position (x, y)
            plt.text(x, y + 2, f'({x}, {y})', color='black', fontsize=12, ha='center')  # Afficher les coordonnées au-dessus du point
            plt.grid(True)
        
        if mask is not None:
            mask_data = mask["land_sea_mask"].data
            mask_rotated = np.rot90(mask_data)
            if x is not None and y is not None and rx is not None and ry is not None:  # Si on effectue un zoom
                mask_zoomed = mask_rotated[y_min:y_max, x_min:x_max]
                plt.imshow(mask_zoomed, cmap='binary', alpha=0.5, extent=[x_min, x_max, y_min, y_max])
            else:  # Si on affiche l'image complète
                plt.imshow(mask_rotated, cmap='binary', alpha=0.5)
            plt.grid(True)

        plt.title(f"{variable} at time {time_index}")
        
        # Sauvegarder l'image en mémoire
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_str = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close()

        return f"data:image/png;base64,{image_str}"

    except IndexError as e:
        raise ValueError(f"Indexing error: {e}")

def list_images_plot(dataset, variable, start_time, end_time, level=None, forecast=None, mask=None, x=None, y=None, rx=None, ry=None):
    time_indexes = get_list_indexes_of_time(dataset, start_time, end_time)
    images = []
    for time_index in time_indexes:
        images.append(plot_and_save_image(dataset, variable, time_index, x, y, rx, ry, level, forecast, mask))
    return images


def check_conditions_for_plot_forecast_statistics(data):
    required_dims = {'level', 'latitude', 'longitude', 'time', 'forecast'}
    data_dims = set(data.dims)
    
    if not required_dims.issubset(data_dims):
        return False

    if not np.issubdtype(data.dtype, np.number):
        return False
    
    return True


def plot_forecast_statistics(ax, forecast_data, ground_truth_data, time, lat, lon, rx, ry, num_forecasts):
    def valeur(dataset, time, level, lat_slice, lon_slice, forecast=None):
        """Extraire les valeurs moyennées sur une région spécifiée."""
        # Utiliser modulo pour s'assurer que les slices sont dans les bornes valides
        lat_slice = [s % lat_dim for s in lat_slice]
        lon_slice = [s % lon_dim for s in lon_slice]
        
        region = dataset.isel(time=time, level=level).sel(
            latitude=slice(min(lat_slice), max(lat_slice)),
            longitude=slice(min(lon_slice), max(lon_slice))
        )
        if forecast is not None:
            region = region.isel(forecast=forecast)
        return region.mean(dim=('latitude', 'longitude')).values

    list_level = forecast_data.level.values

    # Récupérer les dimensions de latitude et longitude
    lat_dim = len(forecast_data.latitude)
    lon_dim = len(forecast_data.longitude)

    # Normaliser les coordonnées lat et lon pour s'assurer qu'elles sont dans les bornes valides
    lat = lat % lat_dim
    lon = lon % lon_dim

    # Définir les tranches pour latitude et longitude en utilisant modulo pour s'assurer qu'ils sont dans les limites
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
            valeur(forecast_data, time, level, lat_slice, lon_slice, forecast) for forecast in range(num_forecasts)
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
        
        # Ajout de la sécurité pour ne tracer la courbe Ground Truth que si elle existe
        if ground_truth_data is not None:
            try:
                ground_truth_value = valeur(ground_truth_data, time, level, lat_slice, lon_slice)
                Y_ground_truth.append(ground_truth_value)
            except KeyError:
                # Si la variable n'existe pas pour ce niveau, ignorer
                Y_ground_truth.append(None)

    X = list_level
    Y_q05 = np.array(Y_q05)
    Y_q25 = np.array(Y_q25)
    Y_q50 = np.array(Y_q50)
    Y_q75 = np.array(Y_q75)
    Y_q95 = np.array(Y_q95)
    
    # Filtrer les valeurs None dans Y_ground_truth
    Y_ground_truth = np.array([gt for gt in Y_ground_truth if gt is not None])

    # Traçage des courbes de percentiles avec remplissage
    ax.fill_between(X, Y_q05, Y_q25, color='yellow', alpha=0.4, label='Q05 - Q25')
    ax.fill_between(X, Y_q25, Y_q50, color='orange', alpha=0.4, label='Q25 - Q50')
    ax.fill_between(X, Y_q50, Y_q75, color='red', alpha=0.4, label='Q50 - Q75')
    ax.fill_between(X, Y_q75, Y_q95, color='purple', alpha=0.4, label='Q75 - Q95')

    # Traçage des lignes de percentiles pour plus de clarté
    ax.plot(X, Y_q05, color='yellow')
    ax.plot(X, Y_q25, color='orange')
    ax.plot(X, Y_q50, color='red')
    ax.plot(X, Y_q75, color='purple')
    ax.plot(X, Y_q95, color='blue', label='Q95')
    
    # Tracer la courbe du Ground Truth sans remplissage
    if len(Y_ground_truth) > 0:  # Tracer uniquement si la Ground Truth est disponible
        ax.plot(X, Y_ground_truth, label='Ground Truth', color='black', linestyle='-', linewidth=2)

    ax.set_xlabel('Pressure Level')
    ax.set_ylabel('Atmospheric Temperature')
    ax.set_title('Forecast Statistics')
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

    # Callback pour mettre à jour les variables et autres inputs basés sur le fichier sélectionné pour le graphique 1
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

    # Callback pour mettre à jour les variables et autres inputs basés sur le fichier sélectionné pour le graphique 2
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
         State('start-year-input', 'value'),
         State('start-month-input', 'value'),
         State('start-day-input', 'value'),
         State('start-hour-input', 'value'),
         State('end-year-input', 'value'),
         State('end-month-input', 'value'),
         State('end-day-input', 'value'),
         State('end-hour-input', 'value'),
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
                                   file1, variable1, start_year, start_month, start_day, start_hour, end_year, end_month, end_day, end_hour,
                                   forecast1, z1, file2, variable2, forecast2, z2, x, y, rx, ry, stored_images1, stored_images2, current_src1, current_src2,
                                   interval_disabled):
        ctx = dash.callback_context
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Si le bouton "Generate" est cliqué, régénérer les images et réinitialiser l'affichage
        if button_id == 'generate-button1':
            if None in [start_year, start_month, start_day, start_hour, end_year, end_month, end_day, end_hour]:
                raise PreventUpdate

            start_time, end_time = get_start_and_end_time(start_year, start_month, start_day, start_hour, end_year, end_month, end_day, end_hour)
            if not start_time or not end_time:
                raise PreventUpdate

            images1, images2 = [], []

            if file1 and variable1:
                try:
                    ds1 = xr.open_zarr(f'{path_repertory}/{file1}')
                except FileNotFoundError:
                    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Dataset not found", interval_disabled

                if not is_period_in_dataset(ds1, start_time, end_time):
                    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Specified time period is not in the dataset", interval_disabled

                level_index1 = list(ds1[variable1].level.values).index(z1) if z1 else None
                images1 = list_images_plot(ds1, variable1, start_time, end_time, level_index1, forecast1, mask, x, y, rx, ry)

            if file2 and variable2:
                try:
                    ds2 = xr.open_zarr(f'{path_repertory}/{file2}')
                except FileNotFoundError:
                    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Dataset not found", interval_disabled

                if not is_period_in_dataset(ds2, start_time, end_time):
                    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Specified time period is not in the dataset", interval_disabled

                level_index2 = list(ds2[variable2].level.values).index(z2) if z2 else None
                images2 = list_images_plot(ds2, variable2, start_time, end_time, level_index2, forecast2, mask, x, y, rx, ry)

            if not images1 and not images2:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "No images generated", interval_disabled

            time_period_text = f"Time Period: {start_time} to {end_time}"

            # Désactiver l'intervalle (lecture automatique) lorsqu'on génère de nouvelles images
            interval_disabled = True

            # Retourner les nouvelles images et réinitialiser l'affichage sur la première image
            return images1, images2, images1[0] if images1 else dash.no_update, images2[0] if images2 else dash.no_update, time_period_text, interval_disabled

        # Si un des boutons "Previous", "Next" ou "Play" est cliqué, mettre à jour les images affichées
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
                interval_disabled = not interval_disabled  # Inverser l'état pour démarrer/arrêter la lecture automatique

            return (
                dash.no_update,
                dash.no_update,
                stored_images1[current_index1] if stored_images1 else dash.no_update,
                stored_images2[current_index2] if stored_images2 else dash.no_update,
                dash.no_update,
                interval_disabled
            )

        raise PreventUpdate


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

            # Vérifiez les conditions pour ajouter l'option "Forecast Statistics"
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
         State('start-year-input', 'value'),
         State('start-month-input', 'value'),
         State('start-day-input', 'value'),
         State('start-hour-input', 'value'),
         State('end-year-input', 'value'),
         State('end-month-input', 'value'),
         State('end-day-input', 'value'),
         State('end-hour-input', 'value'),
         State('range-x', 'value'), 
         State('range-y', 'value'), 
         State('range-rx', 'value'), 
         State('range-ry', 'value'),
         State('metrics-variables-options2', 'value'),  # Sélection de la métrique
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
            start_time, end_time = get_start_and_end_time(
                s_year, s_month, s_day, s_hour,
                e_year, e_month, e_day, e_hour
            )
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
                lat_dim = len(ds_forecast.latitude)
                lon_dim = len(ds_forecast.longitude)

                # Normaliser les coordonnées avec modulo
                x = x % lon_dim
                y = y % lat_dim

                ds_ground_truth = None
                if variable in xr.open_zarr(f'{path_repertory}/{selected_ground_truth}').variables:
                    ds_ground_truth = xr.open_zarr(f'{path_repertory}/{selected_ground_truth}')
            except FileNotFoundError:
                return dash.no_update, dash.no_update

            if not is_period_in_dataset(ds_forecast, start_time, end_time):
                return dash.no_update, dash.no_update

            time_indexes = get_list_indexes_of_time(ds_forecast, start_time, end_time)
            num_forecasts = len(ds_forecast['forecast'])

            images = []
            for time_index in time_indexes:
                buf = BytesIO()
                plt.figure(figsize=(16, 9))  # 16:9 format
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
                    num_forecasts
                )

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

        raise dash.exceptions.PreventUpdate
