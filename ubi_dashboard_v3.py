import streamlit as st
import pandas as pd
import numpy as np
import datetime 
import copy # For deep copying params
import os # For directory scanning
import yaml # For loading scenario files
from ubi import UBISimulatorRealisticV3, DEFAULT_SCENARIO_PARAMS

# --- Formatter Dictionary (align with UBISimulatorRealisticV3 output) ---
formatters_dict = {
    "Total Population (M)": "{:.1f} M",
    "GDP Real (Trillion EUR)": "€{:.2f} T",
    "GDP Growth Rate": "{:.2%}",
    "GDP Potential (Trillion EUR)": "€{:.2f} T",
    "Output Gap (%)": "{:.1f}%",
    "Inflation Rate": "{:.2%}",
    "Price Level Index": "{:.1f}",
    "Capital Stock (Trillion EUR)": "€{:.2f} T",
    "Investment (Trillion EUR)": "€{:.2f} T",
    "TFP Level (Index)": "{:.1f}",
    "TFP Growth Rate": "{:.2%}",
    "Labor Input (Billion Hours)": "{:,.0f} B h",
    "Unemployment Rate": "{:.1%}",
    "Avg Hours Worked": "{:.0f} h",
    "UBI Cost (Billion EUR)": "€{:,.1f} B",
    "UBI Nominal Amount (€)": "€{:,.0f}",
    "Govt Deficit/GDP Ratio": "{:.1%}",
    "Govt Debt/GDP Ratio": "{:.1%}",
    "Gini Coefficient": "{:.3f}",
    "Poverty Rate": "{:.1%}",
    "Births (M)": "{:.2f} M"
}

st.set_page_config(layout="wide")
st.title("Simulador UBI v3 (Integrado com ubi.py)")
st.markdown("Dashboard utilizando `UBISimulatorRealisticV3` de `ubi.py`. Os dados são carregados dinamicamente com base no país selecionado.")

# --- Sidebar para Controles ---
st.sidebar.header("⚙️ Parâmetros da Simulação")

# --- Scenario Loading Logic ---
SCENARIO_DIR = "scenarios"

def get_available_scenarios(scenario_dir=SCENARIO_DIR):
    if not os.path.exists(scenario_dir):
        return []
    return [f for f in os.listdir(scenario_dir) if f.endswith(".yaml") or f.endswith(".yml")]

def load_scenario_params(scenario_filename, scenario_dir=SCENARIO_DIR):
    filepath = os.path.join(scenario_dir, scenario_filename)
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Erro ao carregar cenário '{scenario_filename}': {e}")
        return None

available_scenarios = get_available_scenarios()

# --- UI Parameter Storage ---
# Use session state to store UI parameters to allow updates when a scenario is loaded.
# Initialize if not present.
if "ui_params" not in st.session_state:
    st.session_state.ui_params = copy.deepcopy(DEFAULT_SCENARIO_PARAMS)


# --- Scenario Selection UI ---
st.sidebar.markdown("---")
st.sidebar.subheader("Carregar Cenário")
selected_scenario_file = st.sidebar.selectbox(
    "Carregar Cenário Predefinido/Salvo:",
    options=["Nenhum (usar controles manuais)"] + available_scenarios,
    index=0,
    key="scenario_select",
    help="Selecione um cenário para carregar seus parâmetros nos controles abaixo."
)

if selected_scenario_file != "Nenhum (usar controles manuais)":
    if "last_loaded_scenario" not in st.session_state:
        st.session_state.last_loaded_scenario = ""
    
    # Only load if the button is pressed OR if the selected file changed AND it's not the initial "Nenhum"
    # This aims to prevent reloading when other widgets cause a rerun.
    # A more robust way might involve a dedicated load button or more complex state tracking.
    # For now, loading on selectbox change if it's a new valid scenario.
    if st.sidebar.button("Carregar Cenário Selecionado", key="load_scenario_button") or \
       (selected_scenario_file != st.session_state.last_loaded_scenario and selected_scenario_file != "Nenhum (usar controles manuais)"):
        loaded_params = load_scenario_params(selected_scenario_file)
        if loaded_params:
            st.session_state.ui_params = copy.deepcopy(loaded_params)
            st.session_state.last_loaded_scenario = selected_scenario_file
            st.experimental_rerun() 

# --- UI for Saving Custom Scenarios ---
st.sidebar.markdown("---")
st.sidebar.subheader("Salvar Cenário")
custom_scenario_name_input = st.sidebar.text_input(
    "Nome para Salvar Cenário Atual:", 
    key="custom_scenario_name_input",
    help="Defina um nome para o seu cenário customizado. Será salvo como .yaml."
)

def sanitize_filename(name):
    # Convert to lowercase, replace spaces with underscores, remove special chars
    name = name.lower().replace(" ", "_")
    name = "".join(c for c in name if c.isalnum() or c == '_')
    return f"{name}.yaml"

if st.sidebar.button("Salvar Cenário Atual", key="save_scenario_button"):
    if custom_scenario_name_input:
        # Ensure the scenarios directory exists
        if not os.path.exists(SCENARIO_DIR):
            try:
                os.makedirs(SCENARIO_DIR)
            except OSError as e:
                st.sidebar.error(f"Erro ao criar diretório de cenários: {e}")
                # Stop further processing if directory cannot be created
                # (or handle more gracefully depending on desired behavior)
                # For now, just show error and stop.
                # This part of the code will not be reached if os.makedirs fails.

        sanitized_name = sanitize_filename(custom_scenario_name_input)
        filepath = os.path.join(SCENARIO_DIR, sanitized_name)
        
        # Parameters to save are those currently reflected in the UI (stored in session_state)
        params_to_save = copy.deepcopy(st.session_state.ui_params)
        # Ensure the 'scenario_name' within the params reflects the user's chosen name
        params_to_save['scenario_name'] = custom_scenario_name_input 

        try:
            with open(filepath, 'w') as f:
                yaml.dump(params_to_save, f, sort_keys=False, default_flow_style=False)
            st.sidebar.success(f"Cenário '{custom_scenario_name_input}' salvo como '{sanitized_name}'!")
            # Refresh the list of available scenarios
            # This typically requires a rerun if get_available_scenarios is called at the start of the script.
            # Streamlit should rerun due to widget interaction (button press).
            # Forcing a rerun here can sometimes be useful if the list update is critical immediately.
            st.experimental_rerun() 
        except Exception as e:
            st.sidebar.error(f"Erro ao salvar cenário: {e}")
    else:
        st.sidebar.warning("Por favor, insira um nome para o cenário.")

# Use parameters from session state for UI elements
current_ui_params_values = st.session_state.ui_params
st.sidebar.markdown("---") # Separator before parameter sections


# Function to get available countries by scanning the 'data' directory
def get_available_countries(data_dir="data"):
    try:
        return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    except FileNotFoundError:
        st.warning(f"Diretório de dados '{data_dir}' não encontrado. Usando lista de países padrão.")
        return ["germany", "france"] 
    except Exception as e:
        st.warning(f"Erro ao escanear diretórios de países: {e}. Usando lista padrão.")
        return ["germany", "france"]

available_countries = get_available_countries()
if not available_countries:
    available_countries = ["germany", "france"] 

default_country_name = current_ui_params_values.get("country", DEFAULT_SCENARIO_PARAMS["country"])
if default_country_name not in available_countries:
    default_country_name = available_countries[0] if available_countries else "germany"


with st.sidebar.expander("⏱️ Configuração Principal da Simulação", expanded=True):
    country_selected_from_ui = st.selectbox(
        "País:",
        available_countries,
        index=available_countries.index(default_country_name) if default_country_name in available_countries else 0,
        key="country_select_dashboard_v3" 
    )
    # Update session state if changed
    if current_ui_params_values.get("country") != country_selected_from_ui:
        current_ui_params_values["country"] = country_selected_from_ui
        # If country changes, we might want to reload defaults for that country or a base scenario
        # For now, other parameters will retain their current values in session_state.

    default_sim_years = current_ui_params_values.get("simulation_years", DEFAULT_SCENARIO_PARAMS["simulation_years"])
    sim_start_year_from_ui = st.number_input(
        "Ano Inicial da Simulação",
        min_value=2000, max_value=2100,
        value=int(default_sim_years["start"]), # Use value from session state or default
        step=1, key=f"sim_start_year_dashboard_v3_{country_selected_from_ui}"
    )
    sim_end_year_from_ui = st.number_input(
        "Ano Final da Simulação",
        min_value=sim_start_year_from_ui, max_value=2150,
        value=int(default_sim_years["end"]), # Use value from session state or default
        step=1, key=f"sim_end_year_dashboard_v3_{country_selected_from_ui}"
    )
    if sim_end_year_from_ui < sim_start_year_from_ui:
        st.sidebar.error("O ano final deve ser maior ou igual ao ano inicial.")
        # Reset to defaults from session state if error
        sim_start_year_from_ui = int(default_sim_years["start"]) 
        sim_end_year_from_ui = int(default_sim_years["end"]) 
        
    current_ui_params_values["simulation_years"]["start"] = sim_start_year_from_ui
    current_ui_params_values["simulation_years"]["end"] = sim_end_year_from_ui

with st.sidebar.expander("🔵 Configuração da RBU", expanded=True):
    default_ubi_params = current_ui_params_values.get("ubi", DEFAULT_SCENARIO_PARAMS["ubi"])
    
    current_ui_params_values["ubi"]["annual_amount_eur_real_start"] = st.slider(
        "Valor Anual REAL Inicial RBU (€)", 0, 25000,
        int(default_ubi_params.get("annual_amount_eur_real_start", 14400)),
        500, format="€%.0f", key=f"ubi_amount_dashboard_v3_{country_selected_from_ui}"
    )
    
    ubi_start_value = max(sim_start_year_from_ui, min(sim_end_year_from_ui, int(default_ubi_params.get("start_year", sim_start_year_from_ui + 3))))
    current_ui_params_values["ubi"]["start_year"] = st.slider(
        "Ano de Início da RBU", sim_start_year_from_ui, sim_end_year_from_ui,
        ubi_start_value, step=1, key=f"ubi_start_dashboard_v3_{country_selected_from_ui}"
    )
    current_ui_params_values["ubi"]["fertility_rate_boost_factor"] = st.slider(
        "Aumento Taxa Fertilidade com RBU (fator)", 0.0, 0.1, 
        float(default_ubi_params.get("fertility_rate_boost_factor", 0.02)), 
        0.005, format="%.3f", key=f"ubi_fert_boost_dashboard_v3_{country_selected_from_ui}"
    )

with st.sidebar.expander("💼 Mercado de Trabalho", expanded=True):
    default_labor_params = current_ui_params_values.get("labor_market", DEFAULT_SCENARIO_PARAMS["labor_market"])
    
    # Ensure the nested dictionary for ubi_participation_reduction_factor exists
    if "ubi_participation_reduction_factor" not in current_ui_params_values["labor_market"]:
        current_ui_params_values["labor_market"]["ubi_participation_reduction_factor"] = \
            copy.deepcopy(DEFAULT_SCENARIO_PARAMS["labor_market"]["ubi_participation_reduction_factor"])

    default_part_reduc = default_labor_params.get("ubi_participation_reduction_factor", {"working_age": 0.05, "old_age": 0.10})

    current_ui_params_values["labor_market"]["ubi_participation_reduction_factor"]["working_age"] = st.slider(
        "Redução Participação (Idade Ativa, %)", 0.0, 15.0,
        float(default_part_reduc.get("working_age", 0.05)) * 100,
        0.5, format="%.1f%%", key=f"labor_reduc_active_dashboard_v3_{country_selected_from_ui}"
    ) / 100.0
    
    current_ui_params_values["labor_market"]["ubi_hours_reduction_factor"] = st.slider(
        "Redução Horas Trabalhadas (Idade Ativa, %)", 0.0, 10.0,
        float(default_labor_params.get("ubi_hours_reduction_factor", 0.01)) * 100,
        0.1, format="%.1f%%", key=f"hours_reduc_dashboard_v3_{country_selected_from_ui}"
    ) / 100.0

# Final parameters for simulation are directly from session state, which has been updated
final_params = copy.deepcopy(st.session_state.ui_params)
# Ensure country from selectbox is used if it was the last thing changed
final_params["country"] = country_selected_from_ui


# --- Funções de Cache e Execução ---
def make_hashable(params_dict):
    if isinstance(params_dict, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in params_dict.items()))
    elif isinstance(params_dict, list):
        return tuple(make_hashable(item) for item in params_dict)
    return params_dict

@st.cache_data
def run_simulation_cached(params_hashable):
    def unhash_params(hashed_params):
        if isinstance(hashed_params, tuple):
            try:
                if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in hashed_params):
                    return {item[0]: unhash_params(item[1]) for item in hashed_params}
            except TypeError: pass 
            return [unhash_params(item) for item in hashed_params]
        return hashed_params
        
    params_for_sim = unhash_params(params_hashable)
    
    simulator_instance = UBISimulatorRealisticV3(params=params_for_sim)
    try:
        simulator_instance.run_simulation()
        return simulator_instance
    except Exception as e:
        st.error(f"Erro na execução da simulação: {e}")
        st.expander("Detalhes dos Parâmetros com Erro").write(params_for_sim)
        empty_sim = UBISimulatorRealisticV3(params=params_for_sim)
        empty_sim.results = {"Baseline (Sem RBU)": pd.DataFrame(), "Com RBU": pd.DataFrame()}
        return empty_sim

params_for_cache = make_hashable(final_params)
simulator = run_simulation_cached(params_for_cache)
results = simulator.get_results()

# --- Exibição dos Resultados ---
st.header(f"📊 Resultados da Simulação para {final_params['country'].title()}")

if not isinstance(results, dict) or \
   "Baseline (Sem RBU)" not in results or not isinstance(results["Baseline (Sem RBU)"], pd.DataFrame) or \
   "Com RBU" not in results or not isinstance(results["Com RBU"], pd.DataFrame):
    st.error("Resultados da simulação inválidos ou não foram gerados.")
    st.stop()

baseline_df = results["Baseline (Sem RBU)"]
ubi_df = results["Com RBU"]

sim_years_df_idx = pd.RangeIndex(
    start=final_params["simulation_years"]["start"], 
    stop=final_params["simulation_years"]["end"] + 1, 
    name="Year"
)
if baseline_df.empty:
     baseline_df = pd.DataFrame(index=sim_years_df_idx, columns=list(formatters_dict.keys())).fillna(0.0)
if ubi_df.empty:
     ubi_df = pd.DataFrame(index=sim_years_df_idx, columns=list(formatters_dict.keys())).fillna(0.0)

st.subheader("Tabela Comparativa Anual")
_sim_start = final_params["simulation_years"]["start"]
_sim_end = final_params["simulation_years"]["end"]
_ubi_start = final_params["ubi"]["start_year"]
default_year_table = min(_sim_end, _ubi_start + 10 if _ubi_start <= _sim_end else _sim_end)
default_year_table = int(max(_sim_start, default_year_table))

year_to_display = st.slider(
    "Selecione o ano para a tabela comparativa:",
    min_value=_sim_start, max_value=_sim_end,
    value=default_year_table, step=1,
    key=f"compare_year_slider_dashboard_v3_{country_selected}"
)

summary_df = simulator.get_summary_dataframe(year_to_display) 
if not summary_df.empty:
    valid_formatters = {k: v for k, v in formatters_dict.items() if k in summary_df.columns or k in summary_df.index}
    try:
        st.dataframe(summary_df.style.format(formatter=valid_formatters, na_rep='-'), use_container_width=True)
    except Exception as e:
        st.warning(f"Erro ao formatar tabela de resumo: {e}. Exibindo dados brutos.")
        st.dataframe(summary_df, use_container_width=True)
else:
    st.warning(f"Não há dados disponíveis para a tabela no ano {year_to_display}.")

st.subheader("📈 Gráficos Comparativos ao Longo do Tempo")
main_indicators_to_plot = [
    "Total Population (M)", "GDP Real (Trillion EUR)", "Inflation Rate", 
    "UBI Cost (Billion EUR)", "Govt Debt/GDP Ratio", "Gini Coefficient", "Poverty Rate",
    "Output Gap (%)", "Price Level Index", "TFP Level (Index)"
]

cols = st.columns(2)
col_idx = 0
for indicator in main_indicators_to_plot:
    if indicator in baseline_df.columns and indicator in ubi_df.columns:
        chart_data = pd.DataFrame({
            'Sem RBU': baseline_df[indicator],
            'Com RBU': ubi_df[indicator]
        })
        chart_data.index = chart_data.index.astype(int) # Ensure index is suitable for plotting

        if not chart_data.empty and not chart_data.isnull().all().all():
            current_col = cols[col_idx % 2]
            with current_col:
                st.markdown(f"**{indicator}**")
                st.line_chart(chart_data)
            col_idx += 1

st.subheader("💾 Dados Completos da Simulação")
with st.expander("Ver/Baixar Tabelas de Dados"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cenário Baseline (Sem RBU)**")
        st.dataframe(baseline_df.style.format(formatter=formatters_dict, na_rep='-'))
        st.download_button("Download Baseline CSV", baseline_df.to_csv(index=True).encode('utf-8'),
                           f"baseline_results_{country_selected}.csv", "text/csv", key=f'download-baseline-dash-v3-{country_selected}')
    with col2:
        st.markdown("**Cenário Com RBU**")
        st.dataframe(ubi_df.style.format(formatter=formatters_dict, na_rep='-'))
        st.download_button("Download RBU CSV", ubi_df.to_csv(index=True).encode('utf-8'),
                           f"ubi_results_{country_selected}.csv", "text/csv", key=f'download-ubi-dash-v3-{country_selected}')

st.sidebar.info("Dashboard integrado com UBISimulatorRealisticV3.")
st.caption(f"Executando para o cenário: {final_params.get('scenario_name', 'N/A')}")
