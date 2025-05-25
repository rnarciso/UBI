import streamlit as st
import pandas as pd
import numpy as np
import os 
import yaml # Para carregar config real no futuro
from scipy.optimize import curve_fit # Para calibração real

# --- 1. Estrutura de Configuração (Adicionando Parâmetros de Inflação) ---
DEFAULT_SCENARIO_PARAMS = {
    "scenario_name": "Realistic Structure v0.2 (with Inflation)",
    "country": "germany", # Added country
    "simulation_years": {"start": 2024, "end": 2075},
    "data_paths": { # Placeholder para caminhos de dados reais
        "initial_population": "population_cohorts_2023.csv", # Relative path
        "initial_economy": "economy_2023.csv", # Relative path
        "initial_social": "social_2023.csv", # Relative path
        "mortality_rates": "mortality_rates_proj.csv", # Relative path
        "fertility_rates": "fertility_rates_proj.csv", # Relative path
        "migration_params": "migration_params.yaml", # Relative path
        "prod_function_calibration": "prod_function_params.csv", # Relative path
        "nairu_estimate": "nairu.csv" # Relative path, generic name
    },
    "demographics": {
        "cohort_step": 5, "max_age": 90, "fertility_age_start": 15,
        "fertility_age_end": 49, "working_age_start": 20, "working_age_end": 64,
        "retirement_age": 65,
        "placeholder_mortality_rate_annual": 0.01,
        "placeholder_fertility_rate_per_woman": 1.5,
        "placeholder_birth_sex_ratio": 1.05,
        "placeholder_net_migration_share_of_pop": 0.0015
    },
    "ubi": {
        "annual_amount_eur_real_start": 14400, # Valor real no ano de início
        "start_year": 2028,
        "min_eligible_age": 18,
        "residency_requirement_years": 5,
        "indexation_lag": 1, # Indexa ao IPCA do ano anterior (lag=1)
        "fertility_rate_boost_factor": 0.02 # Exemplo: Aumento de 2% nas taxas ASFR com UBI
    },
    "labor_market": {
        "placeholder_participation_rate_working_age": 0.82,
        "placeholder_participation_rate_old": 0.08,
        "placeholder_avg_hours": 1580,
        "ubi_participation_reduction_factor": {"working_age": 0.05, "old_age": 0.10},
        "ubi_hours_reduction_factor": 0.01,
        "baseline_unemployment_rate": 0.055,
        "nairu": 0.045, # Placeholder NAIRU (Fonte: AMECO/OECD/Estudos)
    },
    "economy": {
        "cobb_douglas_alpha": 0.35,
        "tfp_calibration_factor": 1.0,
        "baseline_tfp_growth_rate": 0.008,
        "ubi_tfp_growth_boost_pp": 0.001,
        "education_lag_years": 15,
        "capital_depreciation_rate": 0.05,
        "investment_rate_gdp": 0.20,
        "baseline_inflation_rate": 0.02, # Inflação alvo/média de longo prazo
        "inflation_output_gap_sensitivity": 0.5, # Quanto inflação reage ao hiato
        "inflation_ubi_demand_sensitivity": 0.3, # Quanto inflação reage ao estímulo UBI
        "inflation_ceiling": 0.10 # Limite superior para evitar divergência no modelo simples
    },
    "social_indicators": {
        "initial_gini": 0.29,
        "initial_poverty_rate": 0.15,
        "ubi_gini_reduction_factor": 0.15,
        "ubi_poverty_reduction_factor": 0.80,
        "poverty_floor": 0.03,
    },
     "government_finance": {
        "initial_debt_gdp_ratio": 0.60,
        "other_gov_spending_share_gdp": 0.25,
        "baseline_tax_revenue_share_gdp": 0.40,
        "ubi_financing_tax_share": 0.7,
        "fiscal_multiplier_transfer": 0.8,
        "fiscal_multiplier_other_gov": 1.2,
        "consumption_propensity_ubi_recipients": 0.8, # Necessário para estímulo
        "consumption_reduction_high_income_financing": 0.2 # Necessário para estímulo
         # ... modelo de evasão fiscal ...
    }
}

# --- Funções Modulares (Incluindo novas para Potencial e Inflação) ---

# (Funções anteriores: get_cohorts, load_initial_population, load_demographic_rates,
# project_population_detailed, load_initial_labor_participation, calculate_labor_input_detailed,
# load_initial_economy, calibrate_economy, calculate_capital_stock, calculate_gdp_cobb_douglas,
# calculate_tfp, calculate_investment, load_initial_social, update_social_indicators,
# calculate_ubi_cost_detailed - podem precisar de pequenos ajustes para passar parâmetros novos)

# <<< Colar aqui as funções do código anterior >>>
# Exemplo de funções que precisam estar aqui (copiar/colar do anterior):
def get_cohorts(params):
    step = params["demographics"]["cohort_step"]
    max_age = params["demographics"]["max_age"]
    return [f"{i}-{i+step-1}" for i in range(0, max_age, step)] + [f"{max_age}+"]

def load_initial_population(country, params):
    filepath = os.path.join("data", country, params['data_paths']['initial_population'])
    st.warning(f"POP: Usando dados placeholder. Carregar de: {filepath}")
    cohorts = get_cohorts(params)
    total_pop = 84.0 # Exemplo
    n_cohorts = len(cohorts)
    dist = {cohorts[i]: total_pop * 2 * (n_cohorts - i) / (n_cohorts * (n_cohorts + 1)) for i in range(n_cohorts)}
    return dist

def load_demographic_rates(country, params):
    """
    Carrega taxas de mortalidade e fertilidade dos arquivos CSV especificados.
    Se a leitura falhar, usa taxas placeholder e emite um aviso.
    Retorna dicionários aninhados: rates[year][cohort] = value
    """
    mortality_path = os.path.join("data", country, params['data_paths']['mortality_rates'])
    fertility_path = os.path.join("data", country, params['data_paths']['fertility_rates'])
    # migration_path = os.path.join("data", country, params['data_paths']['migration_params']) # Added for migration
    mortality_rates_proj = {}
    fertility_rates_proj = {}
    sim_start = params["simulation_years"]["start"]
    sim_end = params["simulation_years"]["end"]
    years_needed = range(sim_start - 1, sim_end + 1) # Inclui ano base
    cohorts = get_cohorts(params)

    data_loaded_successfully = {'mortality': False, 'fertility': False}

    # --- Tentar Carregar Mortalidade ---
    try:
        if os.path.exists(mortality_path):
            df_mort = pd.read_csv(mortality_path)
            # Validação básica das colunas esperadas
            if not {'year', 'cohort', 'mortality_rate_annual'}.issubset(df_mort.columns):
                 raise ValueError(f"Colunas esperadas ausentes no CSV de mortalidade: {mortality_path}")

            # Organiza os dados no formato dicionário aninhado
            df_mort_pivot = df_mort.pivot(index='year', columns='cohort', values='mortality_rate_annual')

            mortality_rates_proj = df_mort_pivot.to_dict('index')

            # Verifica se todos os anos e coortes necessários estão presentes (opcional, mas bom)
            missing_years = [y for y in years_needed if y not in mortality_rates_proj]
            if missing_years:
                 st.warning(f"MORTALIDADE ({country}): Anos ausentes no CSV ({mortality_path}): {missing_years}. Usando último valor conhecido ou fallback.")
                 # Lógica de fallback para anos ausentes (ex: usar valor do último ano)
                 last_valid_year = max(y for y in mortality_rates_proj if y < min(missing_years)) if any(y < min(missing_years) for y in mortality_rates_proj) else None
                 for year in missing_years:
                     mortality_rates_proj[year] = mortality_rates_proj[last_valid_year] if last_valid_year else {} # Ou outra lógica

            # Verifica coortes ausentes em cada ano (e preenche com placeholder se necessário)
            for year in years_needed:
                if year not in mortality_rates_proj: continue # Ano já tratado acima
                for c in cohorts:
                    if c not in mortality_rates_proj[year] or pd.isna(mortality_rates_proj[year][c]):
                        # st.warning(f"MORTALIDADE ({country}): Coorte {c} ausente/NaN no ano {year} em {mortality_path}. Usando placeholder.")
                        # Preenche com placeholder ou último valor conhecido da coorte
                        mortality_rates_proj[year][c] = params["demographics"]["placeholder_mortality_rate_annual"] # Ou lógica melhor

            print(f"INFO ({country}): Dados de mortalidade carregados de: {mortality_path}") # Confirmação (opcional)
            data_loaded_successfully['mortality'] = True
        else:
            raise FileNotFoundError(f"Arquivo não encontrado: {mortality_path}")
    except Exception as e:
        st.warning(f"MORTALIDADE ({country}): Falha ao carregar de '{mortality_path}' ({e}). Usando taxas placeholder.")
        # Lógica Placeholder para MORTALIDADE
        mortality = {c: params["demographics"]["placeholder_mortality_rate_annual"] for c in cohorts}
        for year_needed in years_needed:
            mortality_rates_proj[year_needed] = mortality.copy()
    # Fim da seção Placeholder de Mortalidade (adicionado para clareza)


    # --- Tentar Carregar Fertilidade ---
    try:
        if os.path.exists(fertility_path):
            df_fert = pd.read_csv(fertility_path)
            # Validação básica
            if not {'year', 'cohort', 'fertility_rate_annual_per_woman'}.issubset(df_fert.columns):
                 raise ValueError(f"Colunas esperadas ausentes no CSV de fertilidade: {fertility_path}")

            df_fert_pivot = df_fert.pivot(index='year', columns='cohort', values='fertility_rate_annual_per_woman')
            fertility_rates_proj_loaded = df_fert_pivot.to_dict('index')

            # Preenche o dicionário final garantindo todas as coortes (com 0 para não férteis)
            for year in years_needed:
                 fertility_rates_proj[year] = {c: 0.0 for c in cohorts} # Inicializa com 0
                 if year in fertility_rates_proj_loaded:
                     for cohort, rate in fertility_rates_proj_loaded[year].items():
                          if cohort in fertility_rates_proj[year] and pd.notna(rate):
                              fertility_rates_proj[year][cohort] = rate
                 else:
                      # Ano ausente no CSV - usar último ano válido ou fallback?
                      if year > min(fertility_rates_proj_loaded.keys(), default=float('inf')): # Adicionado default para evitar erro se vazio
                           last_valid_year_fert = max(y for y in fertility_rates_proj_loaded if y < year)
                           fertility_rates_proj[year] = fertility_rates_proj_loaded.get(last_valid_year_fert, {c: 0.0 for c in cohorts}) # Reutiliza taxas, com fallback
                      else: # Se anos iniciais faltarem, não há o que fazer senão erro ou placeholder
                           raise FileNotFoundError(f"Ano inicial {year} ausente em {fertility_path}")

            print(f"INFO ({country}): Dados de fertilidade carregados de: {fertility_path}") # Confirmação (opcional)
            data_loaded_successfully['fertility'] = True
        else:
            raise FileNotFoundError(f"Arquivo não encontrado: {fertility_path}")

    except Exception as e:
        st.warning(f"FERTILIDADE ({country}): Falha ao carregar de '{fertility_path}' ({e}). Usando taxas placeholder.")
        # Lógica Placeholder (se a leitura falhar)
        fertility = {c: 0.0 for c in cohorts}
        step = params["demographics"]["cohort_step"]
        f_start, f_end = params["demographics"]["fertility_age_start"], params["demographics"]["fertility_age_end"]
        tfr = params["demographics"]["placeholder_fertility_rate_per_woman"]
        fertile_cohorts_list = [c for c in cohorts if f_start <= int(c.split('-')[0] if '-' in c else params["demographics"]["max_age"]) <= f_end]
        num_fertile_cohorts = len(fertile_cohorts_list)
        avg_asfr_annual = tfr / (f_end - f_start + 1) if (f_end - f_start + 1) > 0 else 0
        peak_cohort_index = num_fertile_cohorts // 2
        total_weight = 0; weights = {}
        for i, c in enumerate(fertile_cohorts_list): weight = max(0, peak_cohort_index - abs(i - peak_cohort_index) + 1); weights[c] = weight; total_weight += weight
        if total_weight > 0:
            for c in fertile_cohorts_list: fertility[c] = avg_asfr_annual * (weights[c] * num_fertile_cohorts / total_weight)
        fertility_reduction_factor_annual = 0.998 # Consider making this a param
        current_fertility = fertility.copy()
        for year_needed in years_needed: # Changed 'year' to 'year_needed' for clarity
             fertility_rates_proj[year_needed] = current_fertility.copy()
             current_fertility = {c: max(0.0001, rate * fertility_reduction_factor_annual) for c, rate in current_fertility.items()}

    # Remove o aviso genérico se AMBOS foram carregados com sucesso
    if data_loaded_successfully['mortality'] and data_loaded_successfully['fertility']:
        pass
    elif not data_loaded_successfully['mortality'] or not data_loaded_successfully['fertility']:
         st.warning(f"DEMOG RATES ({country}): Usando taxas placeholder para Mortalidade e/ou Fertilidade devido a falha no carregamento.")

    # TODO: Load migration_params.yaml here
    # For now, returning empty dict for migration_params
    migration_params = {}
    try:
        migration_path = os.path.join("data", country, params['data_paths']['migration_params'])
        if os.path.exists(migration_path):
            with open(migration_path, 'r') as f:
                migration_params = yaml.safe_load(f)
            print(f"INFO ({country}): Parâmetros de migração carregados de: {migration_path}")
        else:
            st.warning(f"MIGRATION ({country}): Arquivo não encontrado: {migration_path}. Usando placeholders ou desativando migração complexa.")
            migration_params = {"type": "placeholder_net_migration_share", "value": params["demographics"]["placeholder_net_migration_share_of_pop"]} # Fallback
    except Exception as e:
        st.warning(f"MIGRATION ({country}): Falha ao carregar de '{migration_path}' ({e}). Usando placeholders.")
        migration_params = {"type": "placeholder_net_migration_share", "value": params["demographics"]["placeholder_net_migration_share_of_pop"]} # Fallback


    return mortality_rates_proj, fertility_rates_proj, migration_params # Return migration_params

def project_population_detailed(prev_pop_dist, mortality_rates_yr, fertility_rates_yr, migration_params_yr, params): # Added migration_params_yr
    """
    Projeta população com envelhecimento, mortes, nascimentos e migração.
    Assume que mortality_rates e fertility_rates são taxas ANUAIS.
    migration_params_yr contém parâmetros para calcular a migração líquida total para o ano.
    """
    cohorts = get_cohorts(params)
    step = params["demographics"]["cohort_step"]
    new_pop_dist = {c: 0.0 for c in cohorts}
    total_prev_pop = sum(prev_pop_dist.values())

    # 1. Nascimentos e Sobrevivência -> População Final 0-4
    total_births = 0
    f_start_age = params["demographics"]["fertility_age_start"]
    f_end_age = params["demographics"]["fertility_age_end"]
    for cohort in cohorts:
        is_last_cohort = (cohort == cohorts[-1])
        try:
            if is_last_cohort: age_min, age_max = params["demographics"]["max_age"], 120
            else: age_min, age_max = map(int, cohort.split('-'))
        except ValueError: age_min, age_max = params["demographics"]["max_age"], 120

        if age_min <= f_end_age and age_max >= f_start_age:
             rate = fertility_rates_yr.get(cohort, 0.0)
             women_in_cohort = prev_pop_dist.get(cohort, 0) / 2
             cohort_births = women_in_cohort * rate * step
             total_births += cohort_births
    mortality_rate_0_4 = mortality_rates_yr.get(cohorts[0], 0.01) # Placeholder se ausente
    mortality_rate_0_4 = max(0, min(1, mortality_rate_0_4))
    survival_rate_step_0_4 = (1 - mortality_rate_0_4) ** step
    new_pop_dist[cohorts[0]] = total_births * survival_rate_step_0_4

    # 2. Envelhecimento e Mortes para Coortes 1 em diante
    for i in range(1, len(cohorts)):
        current_cohort = cohorts[i]
        prev_cohort = cohorts[i-1]
        pop_start_prev_cohort = prev_pop_dist.get(prev_cohort, 0)
        mortality_rate_prev_cohort = mortality_rates_yr.get(prev_cohort, 0.01) # Placeholder
        mortality_rate_prev_cohort = max(0, min(1, mortality_rate_prev_cohort))
        survival_rate_step_prev = (1 - mortality_rate_prev_cohort) ** step
        pop_aging_into_current = pop_start_prev_cohort * survival_rate_step_prev

        if current_cohort == cohorts[-1]:
            pop_start_last_cohort = prev_pop_dist.get(current_cohort, 0)
            mortality_rate_last = mortality_rates_yr.get(current_cohort, 0.15) # Placeholder
            mortality_rate_last = max(0, min(1, mortality_rate_last))
            survival_rate_step_last = (1 - mortality_rate_last) ** step
            survivors_within_last = pop_start_last_cohort * survival_rate_step_last
            new_pop_dist[current_cohort] = pop_aging_into_current + survivors_within_last
        else:
            new_pop_dist[current_cohort] = pop_aging_into_current

    # 3. Migração (aplicada ao final, sobre a distribuição resultante)
    # Determina a migração líquida total para o ano atual
    migration_net_total = 0
    mig_type = migration_params_yr.get("type", "placeholder_net_migration_share")
    if mig_type == "placeholder_net_migration_share":
        migration_net_total = total_prev_pop * migration_params_yr.get("value", 0.0015) # Usa valor do YAML ou default
    elif mig_type == "fixed_number_per_year":
        migration_net_total = migration_params_yr.get("value", 0) # Usa valor do YAML ou default
    # Adicionar mais tipos de modelos de migração aqui (ex: por coorte, econômico)

    current_total_pop_before_migration = sum(new_pop_dist.values())
    if current_total_pop_before_migration > 0:
         for cohort in cohorts:
              # Distribuição proporcional (pode ser melhorada com dados de perfil etário de migrantes)
              proportion = new_pop_dist.get(cohort, 0) / current_total_pop_before_migration
              migration_cohort = migration_net_total * proportion
              new_pop_dist[cohort] = max(0, new_pop_dist.get(cohort, 0) + migration_cohort)
    elif migration_net_total > 0: # Caso a população seja 0, mas haja entrada de migrantes
        migration_per_cohort = migration_net_total / len(cohorts)
        for cohort in cohorts: new_pop_dist[cohort] = max(0, new_pop_dist.get(cohort, 0) + migration_per_cohort)

    for cohort in cohorts: new_pop_dist[cohort] = max(0, new_pop_dist.get(cohort, 0))
    return new_pop_dist, total_births # total_births já está em unidades (não milhões)

def load_initial_labor_participation(country, params):
     filepath = os.path.join("data", country, "labor_participation_rates.csv") # Assuming a CSV for this
     st.warning(f"PART RATE ({country}): Usando taxas placeholder. Idealmente carregar de: {filepath}")
     # TODO: Implement actual CSV loading for participation rates per cohort
     cohorts = get_cohorts(params)
     rates = {c: 0.0 for c in cohorts}
     w_start, w_end = params["demographics"]["working_age_start"], params["demographics"]["working_age_end"]
     step = params["demographics"]["cohort_step"]
     for i_age in range(w_start, w_end + 1, step): # Renamed loop variable
         cohort_name = f"{i_age}-{i_age+step-1}" # Renamed variable
         if cohort_name in rates: rates[cohort_name] = params["labor_market"]["placeholder_participation_rate_working_age"]
     for i_age in range(params["demographics"]["retirement_age"], params["demographics"]["max_age"] + 1, step): # Renamed loop variable
          if i_age == params["demographics"]["max_age"]: cohort_name = f"{i_age}+" # Renamed variable
          else: cohort_name = f"{i_age}-{i_age+step-1}" # Renamed variable
          if cohort_name in rates: rates[cohort_name] = params["labor_market"]["placeholder_participation_rate_old"]
     return rates

def calculate_labor_input_detailed(pop_dist, participation_rates, avg_hours, unemployment_rate, ubi_impact_participation, ubi_impact_hours, ubi_active, params):
    total_hours = 0
    w_start, w_end = params["demographics"]["working_age_start"], params["demographics"]["working_age_end"]
    retire_age = params["demographics"]["retirement_age"]
    for cohort_name, pop_val in pop_dist.items(): # Renamed variables
        try: age_min = int(cohort_name.split('-')[0])
        except: age_min = params["demographics"]["max_age"]
        base_rate = participation_rates.get(cohort_name, 0)
        ubi_effect_key = None
        if w_start <= age_min <= w_end: ubi_effect_key = "working_age"
        elif age_min >= retire_age: ubi_effect_key = "old_age"
        rate_reduction = ubi_impact_participation.get(ubi_effect_key, 0) if ubi_active and ubi_effect_key else 0
        hours_reduction = ubi_impact_hours if ubi_active and ubi_effect_key == "working_age" else 0 # Assuming hours reduction only for working age
        current_part_rate = base_rate * (1 - rate_reduction)
        employed = pop_val * current_part_rate * (1 - unemployment_rate) # pop_val already in millions
        current_hours = avg_hours * (1 - hours_reduction)
        total_hours += employed * current_hours # result is Millions * Hours -> keeping units consistent with GDP calculation needs
    return total_hours * 1_000_000 # Convert final result to actual hours for GDP calc.

def load_initial_economy(country, params):
    """Carrega dados econômicos iniciais do CSV."""
    filepath = os.path.join("data", country, params['data_paths']['initial_economy'])
    try:
        if os.path.exists(filepath):
            df_econ = pd.read_csv(filepath)
            # Assume que o CSV tem uma linha com os valores para o ano base
            # e colunas como 'GDP', 'Capital', 'TFP', 'PriceLevel'
            # Exemplo: GDP_trillion_eur, Capital_trillion_eur, TFP_index, PriceLevel_index
            initial_econ_data = df_econ.iloc[0].to_dict()
            st.success(f"ECON ({country}): Dados econômicos carregados de: {filepath}")
            return {
                "GDP": initial_econ_data.get("GDP_trillion_eur", 4.0), # Default if column missing
                "Capital": initial_econ_data.get("Capital_trillion_eur", 11.2), # Default
                "TFP": initial_econ_data.get("TFP_index", 100.0), # Default
                "PriceLevel": initial_econ_data.get("PriceLevel_index", 100.0) # Default
            }
        else:
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    except Exception as e:
        st.warning(f"ECON ({country}): Falha ao carregar de '{filepath}' ({e}). Usando dados placeholder.")
        return {
            "GDP": 4.0, "Capital": 11.2, "TFP": 100.0, "PriceLevel": 100.0
        }


def calibrate_economy(country, params): # Added country
     filepath_calib = os.path.join("data", country, params['data_paths']['prod_function_calibration'])
     st.warning(f"CALIB ({country}): Calibração placeholder. Idealmente carregar de: {filepath_calib}")
     # TODO: Implement loading of calibration params (alpha, tfp_factor) from filepath_calib
     return params["economy"]["cobb_douglas_alpha"], params["economy"]["tfp_calibration_factor"]

def calculate_capital_stock(prev_capital, investment_trillions, depreciation_rate):
    return prev_capital * (1 - depreciation_rate) + investment_trillions

def calculate_gdp_cobb_douglas(tfp_level, capital_stock, labor_input_hours, alpha, scale_factor):
    if capital_stock <= 0 or labor_input_hours <= 0 or tfp_level <= 0: return 0
    # scale_factor deve ser calibrado offline para TFP=100 e K, L iniciais resultarem em Y inicial
    # Labor input is in actual hours here. GDP will be in Trillions if K is Trillions.
    gdp = scale_factor * tfp_level * (capital_stock**alpha) * ((labor_input_hours / 1e9)**(1-alpha)) # Labor in billions for typical scale factors
    return gdp

def calculate_tfp(prev_tfp, base_growth, ubi_boost, edu_effect):
     return prev_tfp * (1 + base_growth + ubi_boost) # Ignora edu_effect

def calculate_investment(gdp, rate):
     return gdp * rate

def load_initial_social(country, params): # Added country
    filepath = os.path.join("data", country, params['data_paths']['initial_social'])
    try:
        if os.path.exists(filepath):
            df_social = pd.read_csv(filepath)
            # Assume CSV tem 'Gini_coefficient' e 'Poverty_rate'
            initial_social_data = df_social.iloc[0].to_dict()
            st.success(f"SOCIAL ({country}): Dados sociais carregados de: {filepath}")
            return {
                "Gini": initial_social_data.get("Gini_coefficient", params["social_indicators"]["initial_gini"]),
                "Poverty": initial_social_data.get("Poverty_rate", params["social_indicators"]["initial_poverty_rate"])
            }
        else:
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    except Exception as e:
        st.warning(f"SOCIAL ({country}): Falha ao carregar de '{filepath}' ({e}). Usando dados placeholder.")
        return {"Gini": params["social_indicators"]["initial_gini"],
                "Poverty": params["social_indicators"]["initial_poverty_rate"]}


def update_social_indicators(prev_indicators, ubi_active, params):
    new_indicators = prev_indicators.copy()
    if ubi_active:
         gini_reduction = params["social_indicators"]["ubi_gini_reduction_factor"]
         poverty_reduction = params["social_indicators"]["ubi_poverty_reduction_factor"]
         poverty_floor = params["social_indicators"]["poverty_floor"]
         new_indicators["Gini"] = prev_indicators["Gini"] * (1 - gini_reduction)
         poverty_reduced = prev_indicators["Poverty"] * (1 - poverty_reduction)
         new_indicators["Poverty"] = max(poverty_floor, poverty_reduced) # Ensure poverty doesn't go below floor
    return new_indicators

def calculate_ubi_cost_detailed(pop_dist, ubi_amount_nominal, params):
    cost = 0
    min_age = params["ubi"]["min_eligible_age"]
    # TODO: Consider residency_requirement_years from params["ubi"]
    for cohort_name, pop_val in pop_dist.items(): # Renamed variables
         try: age_min = int(cohort_name.split('-')[0])
         except: age_min = params["demographics"]["max_age"]
         if age_min >= min_age: cost += pop_val * ubi_amount_nominal # pop_val is in millions, ubi_amount is annual
    return cost # Cost is already in Millions * EUR = Millions EUR. Convert to Billions later.


def calculate_fiscal_balance(gdp_real, price_level_index, gdp_real_prev, price_level_prev, ubi_cost_m, prev_debt_ratio, params): # ubi_cost_m in Millions
     """Calcula balanço fiscal e nova dívida/PIB (nominal)."""
     gdp_nominal_t = gdp_real * (price_level_index / 100) if price_level_index else gdp_real
     gdp_nominal_prev = gdp_real_prev * (price_level_prev / 100) if price_level_prev else gdp_real_prev

     ubi_cost_b = ubi_cost_m / 1000 # Convert UBI cost from Millions to Billions for consistency with other fiscal items

     other_spending_b = (params["government_finance"]["other_gov_spending_share_gdp"] * gdp_nominal_t) # Already in Trillions, convert to Billions
     baseline_revenue_b = (params["government_finance"]["baseline_tax_revenue_share_gdp"] * gdp_nominal_t) # Already in Trillions, convert to Billions

     # Ensure conversion to billions for calculations if they are indeed shares of GDP (Trillions)
     other_spending_b *= 1000
     baseline_revenue_b *= 1000

     ubi_tax_coverage = params["government_finance"]["ubi_financing_tax_share"]
     additional_tax_b = ubi_cost_b * ubi_tax_coverage # Billions
     # ubi_deficit_b = ubi_cost_b * (1 - ubi_tax_coverage) # Billions - This is part of total deficit

     total_spending_b = other_spending_b + ubi_cost_b # All in Billions
     total_revenue_b = baseline_revenue_b + additional_tax_b # All in Billions

     deficit_b = total_spending_b - total_revenue_b # Billions
     deficit_gdp_ratio = (deficit_b / 1000) / gdp_nominal_t if gdp_nominal_t > 0 else 0 # Deficit in Trillions / GDP in Trillions

     prev_debt_nominal_t = prev_debt_ratio * gdp_nominal_prev if gdp_nominal_prev is not None and gdp_nominal_prev > 0 else 0 # Prev Debt in Trillions
     current_debt_nominal_t = prev_debt_nominal_t + (deficit_b / 1000) # Current Debt in Trillions
     new_debt_ratio = current_debt_nominal_t / gdp_nominal_t if gdp_nominal_t > 0 else prev_debt_ratio

     return deficit_gdp_ratio, new_debt_ratio


# --- Novas Funções para PIB Potencial e Inflação ---
def calculate_potential_gdp(tfp_level, capital_stock, pop_dist, base_participation_rates, avg_hours, nairu, alpha, scale_factor, params): # Added base_participation_rates
    """Calcula o PIB Potencial usando NAIRU e taxas de participação base."""
    potential_labor_input_actual_hours = 0
    w_start, w_end = params["demographics"]["working_age_start"], params["demographics"]["working_age_end"]
    retire_age = params["demographics"]["retirement_age"]
    for cohort_name, pop_val in pop_dist.items():
         base_rate = base_participation_rates.get(cohort_name, 0) # Use loaded base rates
         try: age_min = int(cohort_name.split('-')[0])
         except: age_min = params["demographics"]["max_age"]

         # Consider only relevant age groups for potential labor force
         if not (w_start <= age_min <= w_end or age_min >= retire_age): continue

         employed_potential = pop_val * base_rate * (1 - nairu) # pop_val in Millions
         potential_labor_input_actual_hours += employed_potential * avg_hours # Millions * Hours

    if potential_labor_input_actual_hours <= 0: return 0
    potential_labor_input_actual_hours *= 1_000_000 # Convert to actual hours

    potential_gdp = calculate_gdp_cobb_douglas(tfp_level, capital_stock, potential_labor_input_actual_hours, alpha, scale_factor)
    return potential_gdp

def calculate_inflation(baseline_inflation, gdp_real, gdp_potential, ubi_cost_m, financing_params, econ_params, gdp_nominal_potential_t): # ubi_cost_m in Millions
    """Calcula a taxa de inflação (simplificado)."""
    if gdp_potential <= 0 or gdp_nominal_potential_t <=0: return baseline_inflation # Use nominal potential GDP for stimulus ratio

    output_gap = (gdp_real - gdp_potential) / gdp_potential if gdp_potential else 0

    prop_consume = financing_params["consumption_propensity_ubi_recipients"]
    tax_share = financing_params["ubi_financing_tax_share"]
    prop_reduce_consume_tax = financing_params["consumption_reduction_high_income_financing"]

    ubi_cost_t = ubi_cost_m / 1_000_000 # Convert UBI cost from Millions EUR to Trillions EUR for ratio with GDP Potential (Trillions)

    net_demand_stimulus_t = (ubi_cost_t * prop_consume) - (ubi_cost_t * tax_share * prop_reduce_consume_tax)
    demand_stimulus_ratio = net_demand_stimulus_t / gdp_nominal_potential_t if gdp_nominal_potential_t > 0 else 0


    gap_sensitivity = econ_params["inflation_output_gap_sensitivity"]
    demand_sensitivity = econ_params["inflation_ubi_demand_sensitivity"]
    gap_effect = gap_sensitivity * max(0, output_gap)
    ubi_effect = demand_sensitivity * demand_stimulus_ratio # Simplified: direct effect of stimulus ratio
    # Consider non-linearity: ubi_effect *= (1 / (1 + max(0, -output_gap*2))) # Amplifies if output gap is small/negative

    inflation = baseline_inflation + gap_effect + ubi_effect
    inflation = min(inflation, econ_params.get("inflation_ceiling", 0.10))
    return max(0, inflation)

def load_nairu_data(country, params):
    filepath = os.path.join("data", country, params['data_paths']['nairu_estimate'])
    try:
        if os.path.exists(filepath):
            # Assuming CSV with 'year' and 'nairu' columns
            df_nairu = pd.read_csv(filepath)
            # For simplicity, let's assume we use a single NAIRU value for the simulation period,
            # e.g., the latest available or an average. Or it could be a time series.
            # Here, taking the last value as a placeholder.
            nairu_value = df_nairu['nairu'].iloc[-1]
            st.success(f"NAIRU ({country}): Dados de NAIRU carregados de: {filepath}. Usando valor: {nairu_value:.3f}")
            return nairu_value
        else:
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    except Exception as e:
        st.warning(f"NAIRU ({country}): Falha ao carregar de '{filepath}' ({e}). Usando placeholder: {params['labor_market']['nairu']}.")
        return params['labor_market']['nairu'] # Fallback to default param


# --- Classe Simulador Principal (Atualizada) ---
class UBISimulatorRealisticV3:
    def __init__(self, scenario_params):
        self.params = scenario_params
        self.country = self.params["country"] # Get country
        sim_years = self.params["simulation_years"]
        self.start_year = sim_years["start"]
        self.end_year = sim_years["end"]
        self.years = np.arange(self.start_year, self.end_year + 1)
        self.results = {}
        self.cohorts = get_cohorts(self.params)

        # Carregar e calibrar
        self.initial_pop_dist = load_initial_population(self.country, self.params)
        self.mortality_rates, self.fertility_rates, self.migration_params_annual = load_demographic_rates(self.country, self.params) # Unpack migration_params
        self.initial_participation_rates = load_initial_labor_participation(self.country, self.params) # Usado para Potencial e base
        self.initial_economy = load_initial_economy(self.country, self.params)
        self.initial_social = load_initial_social(self.country, self.params)
        self.alpha, self.tfp_factor = calibrate_economy(self.country, self.params)
        self.nairu = load_nairu_data(self.country, self.params) # Load NAIRU

    def _initialize_scenario_dataframe(self):
        columns = [
            "Total Population (M)", "GDP Real (Trillion EUR)", "GDP Growth Rate",
            "GDP Potential (Trillion EUR)", "Output Gap (%)",
            "Inflation Rate", "Price Level Index",
            "Capital Stock (Trillion EUR)", "Investment (Trillion EUR)",
            "TFP Level (Index)", "TFP Growth Rate",
            "Labor Input (Billion Hours)", "Unemployment Rate", "Avg Hours Worked", # Labor Input in Billions for display
            "UBI Cost (Billion EUR)", "UBI Nominal Amount (€)", # UBI Cost in Billions for display
            "Govt Deficit/GDP Ratio", "Govt Debt/GDP Ratio",
            "Gini Coefficient", "Poverty Rate", "Births (M)" # Births in Millions
        ]
        for cohort_name in self.cohorts: columns.extend([f"Pop {cohort_name} (M)", f"Part Rate {cohort_name}"]) # Use consistent naming
        df = pd.DataFrame(index=self.years, columns=columns, dtype=float)
        base_year = self.start_year - 1

        df.loc[base_year, "Total Population (M)"] = sum(self.initial_pop_dist.values())
        for cohort_name, pop_val in self.initial_pop_dist.items(): df.loc[base_year, f"Pop {cohort_name} (M)"] = pop_val
        for cohort_name, rate_val in self.initial_participation_rates.items(): df.loc[base_year, f"Part Rate {cohort_name}"] = rate_val # Use initial_participation_rates
        df.loc[base_year, "GDP Real (Trillion EUR)"] = self.initial_economy["GDP"]
        df.loc[base_year, "Capital Stock (Trillion EUR)"] = self.initial_economy["Capital"]
        df.loc[base_year, "TFP Level (Index)"] = self.initial_economy["TFP"]
        df.loc[base_year, "Price Level Index"] = self.initial_economy["PriceLevel"]
        df.loc[base_year, "Inflation Rate"] = self.params["economy"]["baseline_inflation_rate"]
        df.loc[base_year, "Unemployment Rate"] = self.nairu # Initialize unemployment with NAIRU for base year consistency
        df.loc[base_year, "Avg Hours Worked"] = self.params["labor_market"]["placeholder_avg_hours"]
        df.loc[base_year, "Gini Coefficient"] = self.initial_social["Gini"]
        df.loc[base_year, "Poverty Rate"] = self.initial_social["Poverty"]
        df.loc[base_year, "Govt Debt/GDP Ratio"] = self.params["government_finance"]["initial_debt_gdp_ratio"]
        df.loc[base_year, "Births (M)"] = 0 # Placeholder for base year
        df.loc[base_year, "UBI Cost (Billion EUR)"] = 0
        df.loc[base_year, "UBI Nominal Amount (€)"] = self.params["ubi"]["annual_amount_eur_real_start"]
        df.loc[base_year, "Govt Deficit/GDP Ratio"] = 0

        gdp_pot_init = calculate_potential_gdp(
            df.loc[base_year, "TFP Level (Index)"], df.loc[base_year, "Capital Stock (Trillion EUR)"],
            self.initial_pop_dist, self.initial_participation_rates, df.loc[base_year, "Avg Hours Worked"],
            self.nairu, self.alpha, self.tfp_factor, self.params
        )
        df.loc[base_year, "GDP Potential (Trillion EUR)"] = gdp_pot_init
        output_gap_init = (self.initial_economy["GDP"] - gdp_pot_init) / gdp_pot_init if gdp_pot_init and gdp_pot_init > 0 else 0
        df.loc[base_year, "Output Gap (%)"] = output_gap_init * 100
        return df


    def run_simulation(self):
        df_baseline = self._initialize_scenario_dataframe()
        df_ubi = self._initialize_scenario_dataframe() # Separate df for UBI scenario

        # Ensure mortality and fertility rates are dictionaries for .get() method
        # If they are not, it implies an issue with how they are loaded or structured.
        # For example, they should be dicts like: {year: {cohort: rate}}
        if not isinstance(self.mortality_rates, dict):
            st.error(f"Mortality rates for {self.country} are not in the expected dictionary format.")
            # Potentially use a placeholder or stop simulation
            # For now, let's assume they are loaded correctly as dicts or this error will stop it.
            return
        if not isinstance(self.fertility_rates, dict):
            st.error(f"Fertility rates for {self.country} are not in the expected dictionary format.")
            return


        for year in self.years:
            prev_year = year - 1

            # --- Baseline Scenario ---
            prev_pop_dist_base = {c: df_baseline.loc[prev_year, f"Pop {c} (M)"] for c in self.cohorts}
            current_migration_params_base = self.migration_params_annual # This should be a dict {type: ..., value: ...} or similar
            
            # Get mortality/fertility for the current year, with fallback to previous year's data or an empty dict
            mortality_rates_yr_base = self.mortality_rates.get(year, self.mortality_rates.get(prev_year, {}))
            fertility_rates_yr_base = self.fertility_rates.get(year, self.fertility_rates.get(prev_year, {}))

            current_pop_dist_base, births_base_m = project_population_detailed(
                prev_pop_dist_base, mortality_rates_yr_base,
                fertility_rates_yr_base, current_migration_params_base, self.params
            )
            df_baseline.loc[year, "Births (M)"] = births_base_m / 1e6 # Store in Millions
            for cohort, pop in current_pop_dist_base.items(): df_baseline.loc[year, f"Pop {cohort} (M)"] = pop
            df_baseline.loc[year, "Total Population (M)"] = sum(current_pop_dist_base.values())

            current_part_rates_base = {c: df_baseline.loc[prev_year, f"Part Rate {c}"] for c in self.cohorts}
            for c, r in current_part_rates_base.items(): df_baseline.loc[year, f"Part Rate {c}"] = r
            df_baseline.loc[year, "Unemployment Rate"] = self.nairu
            df_baseline.loc[year, "Avg Hours Worked"] = df_baseline.loc[prev_year, "Avg Hours Worked"]

            labor_input_base_actual_hours = calculate_labor_input_detailed(current_pop_dist_base, current_part_rates_base,
                                                                 df_baseline.loc[year, "Avg Hours Worked"], self.nairu,
                                                                 {}, 0, False, self.params)
            df_baseline.loc[year, "Labor Input (Billion Hours)"] = labor_input_base_actual_hours / 1e9

            tfp_base = calculate_tfp(df_baseline.loc[prev_year, "TFP Level (Index)"], self.params["economy"]["baseline_tfp_growth_rate"], 0, 0)
            df_baseline.loc[year, "TFP Level (Index)"] = tfp_base
            df_baseline.loc[year, "TFP Growth Rate"] = (tfp_base / df_baseline.loc[prev_year, "TFP Level (Index)"] - 1) if df_baseline.loc[prev_year, "TFP Level (Index)"] and df_baseline.loc[prev_year, "TFP Level (Index)"] != 0 else 0

            gdp_prev_base = df_baseline.loc[prev_year, "GDP Real (Trillion EUR)"]
            investment_base = calculate_investment(gdp_prev_base, self.params["economy"]["investment_rate_gdp"])
            df_baseline.loc[year, "Investment (Trillion EUR)"] = investment_base
            capital_stock_base = calculate_capital_stock(df_baseline.loc[prev_year, "Capital Stock (Trillion EUR)"], investment_base, self.params["economy"]["capital_depreciation_rate"])
            df_baseline.loc[year, "Capital Stock (Trillion EUR)"] = capital_stock_base

            gdp_base = calculate_gdp_cobb_douglas(tfp_base, capital_stock_base, labor_input_base_actual_hours, self.alpha, self.tfp_factor)
            df_baseline.loc[year, "GDP Real (Trillion EUR)"] = gdp_base
            df_baseline.loc[year, "GDP Growth Rate"] = (gdp_base / gdp_prev_base - 1) if gdp_prev_base and gdp_prev_base > 0 else 0

            gdp_potential_base = calculate_potential_gdp(tfp_base, capital_stock_base, current_pop_dist_base,
                                                         self.initial_participation_rates, 
                                                         df_baseline.loc[year, "Avg Hours Worked"], self.nairu,
                                                         self.alpha, self.tfp_factor, self.params)
            df_baseline.loc[year, "GDP Potential (Trillion EUR)"] = gdp_potential_base
            output_gap_base = (gdp_base - gdp_potential_base) / gdp_potential_base if gdp_potential_base and gdp_potential_base > 0 else 0
            df_baseline.loc[year, "Output Gap (%)"] = output_gap_base * 100
            
            prev_price_level_base = df_baseline.loc[prev_year, "Price Level Index"] if prev_year in df_baseline.index else 100
            gdp_nominal_potential_base_t = gdp_potential_base * (prev_price_level_base / 100)


            inflation_base = calculate_inflation(self.params["economy"]["baseline_inflation_rate"], gdp_base, gdp_potential_base,
                                                 0, self.params["government_finance"], self.params["economy"], gdp_nominal_potential_base_t)
            df_baseline.loc[year, "Inflation Rate"] = inflation_base
            df_baseline.loc[year, "Price Level Index"] = prev_price_level_base * (1 + inflation_base)

            df_baseline.loc[year, "UBI Cost (Billion EUR)"] = 0
            df_baseline.loc[year, "UBI Nominal Amount (€)"] = 0
            deficit_base, debt_base = calculate_fiscal_balance(gdp_base, df_baseline.loc[year, "Price Level Index"],
                                                               gdp_prev_base, prev_price_level_base, # Use prev_price_level_base
                                                               0, df_baseline.loc[prev_year, "Govt Debt/GDP Ratio"], self.params)
            df_baseline.loc[year, "Govt Deficit/GDP Ratio"] = deficit_base
            df_baseline.loc[year, "Govt Debt/GDP Ratio"] = debt_base
            prev_social_base = {"Gini": df_baseline.loc[prev_year, "Gini Coefficient"], "Poverty": df_baseline.loc[prev_year, "Poverty Rate"]}
            current_social_base = update_social_indicators(prev_social_base, False, self.params)
            df_baseline.loc[year, "Gini Coefficient"] = current_social_base["Gini"]
            df_baseline.loc[year, "Poverty Rate"] = current_social_base["Poverty"]


            # --- UBI Scenario ---
            ubi_active = year >= self.params["ubi"]["start_year"]
            
            mortality_rates_yr_ubi = self.mortality_rates.get(year, self.mortality_rates.get(prev_year, {}))
            fertility_rates_scenario_ubi = self.fertility_rates.get(year, self.fertility_rates.get(prev_year, {}))
            boost = self.params["ubi"].get("fertility_rate_boost_factor", 0.0) if ubi_active else 0.0
            fertility_rates_ubi_yr = {cohort: rate * (1 + boost) for cohort, rate in fertility_rates_scenario_ubi.items()}

            prev_pop_dist_ubi = {c: df_ubi.loc[prev_year, f"Pop {c} (M)"] for c in self.cohorts}
            current_migration_params_ubi = self.migration_params_annual 
            current_pop_dist_ubi, births_ubi_m = project_population_detailed(
                prev_pop_dist_ubi, mortality_rates_yr_ubi,
                fertility_rates_ubi_yr, current_migration_params_ubi, self.params
            )
            df_ubi.loc[year, "Births (M)"] = births_ubi_m / 1e6
            for cohort, pop in current_pop_dist_ubi.items(): df_ubi.loc[year, f"Pop {cohort} (M)"] = pop
            df_ubi.loc[year, "Total Population (M)"] = sum(current_pop_dist_ubi.values())

            prev_ubi_nominal = df_ubi.loc[prev_year, "UBI Nominal Amount (€)"]
            current_ubi_nominal = 0
            if ubi_active:
                if year == self.params["ubi"]["start_year"]:
                    current_ubi_nominal = self.params["ubi"]["annual_amount_eur_real_start"]
                else:
                    lag = self.params["ubi"].get("indexation_lag", 1)
                    inflation_for_indexing = df_ubi.loc[year - lag, "Inflation Rate"] if (year - lag) in df_ubi.index and pd.notna(df_ubi.loc[year - lag, "Inflation Rate"]) else self.params["economy"]["baseline_inflation_rate"]
                    current_ubi_nominal = prev_ubi_nominal * (1 + inflation_for_indexing)
            df_ubi.loc[year, "UBI Nominal Amount (€)"] = current_ubi_nominal

            current_part_rates_ubi = {}
            base_part_rates_for_ubi_scenario = {c: df_ubi.loc[prev_year, f"Part Rate {c}"] for c in self.cohorts}
            for c in self.cohorts:
                base_rate = base_part_rates_for_ubi_scenario[c]
                reduction_config = self.params["labor_market"]["ubi_participation_reduction_factor"]
                age_min_cohort = int(c.split('-')[0]) if '-' in c else self.params["demographics"]["max_age"]
                key = "working_age" if self.params["demographics"]["working_age_start"] <= age_min_cohort <= self.params["demographics"]["working_age_end"] else \
                      "old_age" if age_min_cohort >= self.params["demographics"]["retirement_age"] else None
                reduction = reduction_config.get(key, 0) if ubi_active and key else 0
                current_part_rates_ubi[c] = base_rate * (1 - reduction)
                df_ubi.loc[year, f"Part Rate {c}"] = current_part_rates_ubi[c]

            hours_reduction_factor = self.params["labor_market"]["ubi_hours_reduction_factor"] if ubi_active else 0
            df_ubi.loc[year, "Avg Hours Worked"] = df_ubi.loc[prev_year, "Avg Hours Worked"] * (1 - hours_reduction_factor)
            df_ubi.loc[year, "Unemployment Rate"] = self.nairu 

            labor_input_ubi_actual_hours = calculate_labor_input_detailed(
                current_pop_dist_ubi, current_part_rates_ubi, df_ubi.loc[year, "Avg Hours Worked"],
                df_ubi.loc[year, "Unemployment Rate"], 
                self.params["labor_market"]["ubi_participation_reduction_factor"] if ubi_active else {}, 
                self.params["labor_market"]["ubi_hours_reduction_factor"] if ubi_active else 0,
                ubi_active, self.params
            )
            df_ubi.loc[year, "Labor Input (Billion Hours)"] = labor_input_ubi_actual_hours / 1e9

            ubi_tfp_boost_val = self.params["economy"]["ubi_tfp_growth_boost_pp"] if ubi_active else 0
            tfp_ubi = calculate_tfp(df_ubi.loc[prev_year, "TFP Level (Index)"], self.params["economy"]["baseline_tfp_growth_rate"], ubi_tfp_boost_val, 0)
            df_ubi.loc[year, "TFP Level (Index)"] = tfp_ubi
            df_ubi.loc[year, "TFP Growth Rate"] = (tfp_ubi / df_ubi.loc[prev_year, "TFP Level (Index)"] - 1) if df_ubi.loc[prev_year, "TFP Level (Index)"] and df_ubi.loc[prev_year, "TFP Level (Index)"] != 0 else 0
            
            gdp_prev_ubi = df_ubi.loc[prev_year, "GDP Real (Trillion EUR)"]
            investment_ubi = calculate_investment(gdp_prev_ubi, self.params["economy"]["investment_rate_gdp"])
            df_ubi.loc[year, "Investment (Trillion EUR)"] = investment_ubi
            capital_stock_ubi = calculate_capital_stock(df_ubi.loc[prev_year, "Capital Stock (Trillion EUR)"], investment_ubi, self.params["economy"]["capital_depreciation_rate"])
            df_ubi.loc[year, "Capital Stock (Trillion EUR)"] = capital_stock_ubi

            gdp_ubi = calculate_gdp_cobb_douglas(tfp_ubi, capital_stock_ubi, labor_input_ubi_actual_hours, self.alpha, self.tfp_factor)
            df_ubi.loc[year, "GDP Real (Trillion EUR)"] = gdp_ubi
            df_ubi.loc[year, "GDP Growth Rate"] = (gdp_ubi / gdp_prev_ubi - 1) if gdp_prev_ubi and gdp_prev_ubi > 0 else 0

            gdp_potential_ubi = calculate_potential_gdp(tfp_ubi, capital_stock_ubi, current_pop_dist_ubi,
                                                        self.initial_participation_rates, 
                                                        df_ubi.loc[year, "Avg Hours Worked"], self.nairu, 
                                                        self.alpha, self.tfp_factor, self.params)
            df_ubi.loc[year, "GDP Potential (Trillion EUR)"] = gdp_potential_ubi
            output_gap_ubi = (gdp_ubi - gdp_potential_ubi) / gdp_potential_ubi if gdp_potential_ubi and gdp_potential_ubi > 0 else 0
            df_ubi.loc[year, "Output Gap (%)"] = output_gap_ubi * 100

            ubi_cost_m = calculate_ubi_cost_detailed(current_pop_dist_ubi, current_ubi_nominal, self.params) if ubi_active else 0
            df_ubi.loc[year, "UBI Cost (Billion EUR)"] = ubi_cost_m / 1000

            prev_price_level_ubi = df_ubi.loc[prev_year, "Price Level Index"] if prev_year in df_ubi.index else 100
            gdp_nominal_potential_ubi_t = gdp_potential_ubi * (prev_price_level_ubi / 100)

            inflation_ubi = calculate_inflation(self.params["economy"]["baseline_inflation_rate"], gdp_ubi, gdp_potential_ubi,
                                                ubi_cost_m, self.params["government_finance"], self.params["economy"], gdp_nominal_potential_ubi_t)
            df_ubi.loc[year, "Inflation Rate"] = inflation_ubi
            df_ubi.loc[year, "Price Level Index"] = prev_price_level_ubi * (1 + inflation_ubi)

            deficit_ubi, debt_ubi = calculate_fiscal_balance(gdp_ubi, df_ubi.loc[year, "Price Level Index"],
                                                             gdp_prev_ubi, prev_price_level_ubi, # Use prev_price_level_ubi
                                                             ubi_cost_m, df_ubi.loc[prev_year, "Govt Debt/GDP Ratio"], self.params)
            df_ubi.loc[year, "Govt Deficit/GDP Ratio"] = deficit_ubi
            df_ubi.loc[year, "Govt Debt/GDP Ratio"] = debt_ubi
            prev_social_ubi = {"Gini": df_ubi.loc[prev_year, "Gini Coefficient"], "Poverty": df_ubi.loc[prev_year, "Poverty Rate"]}
            current_social_ubi = update_social_indicators(prev_social_ubi, ubi_active, self.params)
            df_ubi.loc[year, "Gini Coefficient"] = current_social_ubi["Gini"]
            df_ubi.loc[year, "Poverty Rate"] = current_social_ubi["Poverty"]

        self.results["Baseline (Sem RBU)"] = df_baseline.drop(self.start_year - 1).fillna(0)
        self.results["Com RBU"] = df_ubi.drop(self.start_year - 1).fillna(0)


    def get_results(self):
        return self.results

    def get_summary_dataframe(self, year):
        if not self.results or "Baseline (Sem RBU)" not in self.results: return pd.DataFrame()
        target_year = max(self.start_year, min(year, self.end_year))
        if target_year not in self.results["Baseline (Sem RBU)"].index: return pd.DataFrame()

        baseline_data = self.results["Baseline (Sem RBU)"].loc[target_year]
        ubi_data = self.results["Com RBU"].loc[target_year]
        core_indicators = [
            "Total Population (M)", "GDP Real (Trillion EUR)", "GDP Growth Rate",
            "Inflation Rate", "Price Level Index", "UBI Nominal Amount (€)",
             "Govt Debt/GDP Ratio", "Gini Coefficient", "Poverty Rate",
            "UBI Cost (Billion EUR)"
        ]
        summary = pd.DataFrame({
            "Indicador": core_indicators,
            "Sem RBU": [baseline_data.get(ind, 'N/A') for ind in core_indicators],
            "Com RBU": [ubi_data.get(ind, 'N/A') for ind in core_indicators]
        }).set_index("Indicador")

        # Formatação
        # Define a função de formatação
        formatter_lambda = lambda x: f"{x:.2f}" if isinstance(x, (float, np.number)) and pd.notna(x) and abs(x) < 1000 else (f"{x:,.0f}" if isinstance(x, (float, np.number)) and pd.notna(x) else x)

        # Aplica a formatação usando .map() em cada coluna numérica
        for col in ["Sem RBU", "Com RBU"]:
            if col in summary.columns: # Verifica se a coluna existe
                # Converte a coluna para objeto para evitar erros de tipo com .map em alguns casos
                summary[col] = summary[col].astype(object).map(formatter_lambda)

        # --- Mantenha as linhas de formatação específicas (para %, Gini, etc.) aqui ABAIXO ---
        # Exemplo (linhas que você já tinha depois do applymap):
        for ind in ["GDP Growth Rate", "Inflation Rate", "Govt Debt/GDP Ratio", "Poverty Rate"]:
            if ind in summary.index:
                summary.loc[ind] = summary.loc[ind].apply(
                 lambda x: f"{float(str(x).replace(',', '')):.1%}" if isinstance(x, (str, float, int, np.number)) and pd.notna(x) and str(x) not in ['N/A', 'Error'] else x
            )
        if "Gini Coefficient" in summary.index:
            summary.loc["Gini Coefficient"] = summary.loc["Gini Coefficient"].apply(
             lambda x: f"{float(str(x).replace(',', '')):.3f}" if isinstance(x, (str, float, int, np.number)) and pd.notna(x) and str(x) not in ['N/A', 'Error'] else x
         )

        return summary


# --- Interface Streamlit (Atualizada) ---
st.set_page_config(layout="wide")
st.title("Simulador RBU v3 (Inflação + Dinâmica Etária)")
st.warning("Modelo com placeholders, calibração simplificada e modelo básico de inflação.")

# --- Sidebar ---
st.sidebar.header("Parâmetros Chave RBU & Economia")
# Carrega parâmetros base
default_params_for_ui = DEFAULT_SCENARIO_PARAMS.copy() # Use a copy for UI modifications

# Controles UI
country_selected = st.sidebar.selectbox("País:", ["germany", "france"], index=["germany", "france"].index(default_params_for_ui["country"]))
ubi_amount_real_start = st.sidebar.slider("Valor Anual REAL Inicial RBU (€)", 0, 25000, default_params_for_ui["ubi"]["annual_amount_eur_real_start"], 500, key=f"ubi_amount_{country_selected}")
ubi_start_year = st.sidebar.slider("Ano Início RBU", default_params_for_ui["simulation_years"]["start"], default_params_for_ui["simulation_years"]["end"] - 1, default_params_for_ui["ubi"]["start_year"], key=f"ubi_start_{country_selected}")
labor_reduction_active = st.sidebar.slider("Redução Participação (Idade Ativa, %)", 0.0, 15.0, default_params_for_ui["labor_market"]["ubi_participation_reduction_factor"]["working_age"] * 100, 0.5, "%.1f%%", key=f"labor_reduc_{country_selected}")
tfp_boost = st.sidebar.slider("Boost TFP Anual (p.p.)", 0.0, 0.5, default_params_for_ui["economy"]["ubi_tfp_growth_boost_pp"] * 100, 0.05, "%.2f p.p.", key=f"tfp_boost_{country_selected}")
inflation_demand_sens = st.sidebar.slider("Sensibilidade Inflação->Demanda UBI", 0.0, 1.0, default_params_for_ui["economy"]["inflation_ubi_demand_sensitivity"], 0.05, key=f"inf_sens_{country_selected}")

# Atualiza parâmetros
current_params = default_params_for_ui.copy() # Start with a fresh copy
current_params["country"] = country_selected # Set selected country
current_params["ubi"]["annual_amount_eur_real_start"] = ubi_amount_real_start
current_params["ubi"]["start_year"] = ubi_start_year
current_params["labor_market"]["ubi_participation_reduction_factor"]["working_age"] = labor_reduction_active / 100
current_params["economy"]["ubi_tfp_growth_boost_pp"] = tfp_boost / 100
current_params["economy"]["inflation_ubi_demand_sensitivity"] = inflation_demand_sens


# --- Execução e Exibição ---
@st.cache_data # Cache now depends on all params, including country
def run_simulation_cached_v3(params_tuple_items): # Streamlit cache needs hashable input
    params_dict = dict(params_tuple_items) # Convert back to dict
    simulator = UBISimulatorRealisticV3(params_dict)
    simulator.run_simulation()
    return simulator

# Convert dict to a hashable form for caching (tuple of sorted items)
current_params_tuple_items = tuple(sorted(current_params.items(), key=lambda item: str(item[0])))
simulator = run_simulation_cached_v3(current_params_tuple_items) # Pass sorted tuple of items

results = simulator.get_results()

if not results:
    st.error("Falha ao executar simulação.")
else:
    st.header(f"Resultados para {country_selected.title()}") # Title includes country
    baseline_df = results["Baseline (Sem RBU)"]
    ubi_df = results["Com RBU"]

    # Tabela Resumo
    st.subheader("Resumo Anual")
    year_display = st.slider("Ano Tabela:", min_value=current_params["simulation_years"]["start"], max_value=current_params["simulation_years"]["end"], value=current_params["simulation_years"]["end"], key=f"table_year_v3_{country_selected}") # Key includes country
    summary = simulator.get_summary_dataframe(year_display)
    if not summary.empty: st.dataframe(summary, use_container_width=True)
    else: st.warning(f"Não foi possível gerar a tabela para o ano {year_display}.")

    # Gráficos Principais
    st.subheader("Evolução Temporal")
    charts_v3 = ["GDP Real (Trillion EUR)", "Inflation Rate", "Price Level Index", "UBI Nominal Amount (€)", "Govt Debt/GDP Ratio", "Gini Coefficient", "Poverty Rate", "UBI Cost (Billion EUR)"]
    for chart_ind in charts_v3:
         if chart_ind in baseline_df.columns and chart_ind in ubi_df.columns:
             st.markdown(f"**{chart_ind}**")
             if chart_ind == "UBI Nominal Amount (€)": # UBI Nominal Amount is 0 in baseline
                  chart_data = pd.DataFrame({'Com RBU': ubi_df[chart_ind]})
             else:
                  chart_data = pd.DataFrame({'Sem RBU': baseline_df[chart_ind], 'Com RBU': ubi_df[chart_ind]})
             st.line_chart(chart_data)
         else:
             st.warning(f"Indicador '{chart_ind}' não plotado para {country_selected.title()}.")

    # Detalhes Demográficos
    with st.expander(f"Detalhes Demográficos (Coortes) - {country_selected.title()}"): # Expander title includes country
         pop_cols = [f"Pop {c} (M)" for c in simulator.cohorts]
         if all(col in baseline_df.columns for col in pop_cols) and all(col in ubi_df.columns for col in pop_cols):
             pop_df_base = baseline_df[pop_cols]
             pop_df_ubi = ubi_df[pop_cols]
             st.markdown("**População por Coorte - Cenário Base**"); st.line_chart(pop_df_base)
             st.markdown("**População por Coorte - Cenário com RBU**"); st.line_chart(pop_df_ubi)
         else:
            st.warning(f"Dados demográficos por coorte não disponíveis para {country_selected.title()}.")