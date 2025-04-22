import streamlit as st
import pandas as pd
import numpy as np
import os 
import yaml # Para carregar config real no futuro
from scipy.optimize import curve_fit # Para calibração real

# --- 1. Estrutura de Configuração (Adicionando Parâmetros de Inflação) ---
DEFAULT_SCENARIO_PARAMS = {
    "scenario_name": "Realistic Structure v0.2 (with Inflation)",
    "simulation_years": {"start": 2024, "end": 2075},
    "data_paths": { # Placeholder para caminhos de dados reais
        "initial_population": "data/population_cohorts_2023.csv",
        "initial_economy": "data/economy_2023.csv", # Incluiria inflação base histórica
        "initial_social": "data/social_2023.csv",
        "mortality_rates": "data/mortality_rates_proj.csv",
        "fertility_rates": "data/fertility_rates_proj.csv",
        "migration_params": "data/migration_params.yaml",
        "prod_function_calibration": "data/prod_function_params.csv",
        "nairu_estimate": "data/nairu_germany.csv" # Fonte: AMECO, OECD, Estudos
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

def load_initial_population(params):
    st.warning(f"POP: Usando dados placeholder. Carregar de: {params['data_paths']['initial_population']}")
    cohorts = get_cohorts(params)
    total_pop = 84.0 # Exemplo
    n_cohorts = len(cohorts)
    dist = {cohorts[i]: total_pop * 2 * (n_cohorts - i) / (n_cohorts * (n_cohorts + 1)) for i in range(n_cohorts)}
    return dist

def load_demographic_rates(params):
    """
    Carrega taxas de mortalidade e fertilidade dos arquivos CSV especificados.
    Se a leitura falhar, usa taxas placeholder e emite um aviso.
    Retorna dicionários aninhados: rates[year][cohort] = value
    """
    mortality_path = params['data_paths']['mortality_rates']
    fertility_path = params['data_paths']['fertility_rates']
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
                 raise ValueError("Colunas esperadas ausentes no CSV de mortalidade.")

            # Organiza os dados no formato dicionário aninhado
            df_mort_pivot = df_mort.pivot(index='year', columns='cohort', values='mortality_rate_annual')
            mortality_rates_proj = df_mort_pivot.to_dict('index')

            # Verifica se todos os anos e coortes necessários estão presentes (opcional, mas bom)
            missing_years = [y for y in years_needed if y not in mortality_rates_proj]
            if missing_years:
                 st.warning(f"MORTALIDADE: Anos ausentes no CSV: {missing_years}. Usando último valor conhecido ou fallback.")
                 # Lógica de fallback para anos ausentes (ex: usar valor do último ano)
                 last_valid_year = max(y for y in mortality_rates_proj if y < min(missing_years)) if any(y < min(missing_years) for y in mortality_rates_proj) else None
                 for year in missing_years:
                     mortality_rates_proj[year] = mortality_rates_proj[last_valid_year] if last_valid_year else {} # Ou outra lógica

            # Verifica coortes ausentes em cada ano (e preenche com placeholder se necessário)
            for year in years_needed:
                if year not in mortality_rates_proj: continue # Ano já tratado acima
                for c in cohorts:
                    if c not in mortality_rates_proj[year] or pd.isna(mortality_rates_proj[year][c]):
                        # st.warning(f"MORTALIDADE: Coorte {c} ausente/NaN no ano {year}. Usando placeholder.")
                        # Preenche com placeholder ou último valor conhecido da coorte
                        mortality_rates_proj[year][c] = params["demographics"]["placeholder_mortality_rate_annual"] # Ou lógica melhor

            print(f"INFO: Dados de mortalidade carregados de: {mortality_path}") # Confirmação (opcional)
            data_loaded_successfully['mortality'] = True
        else:
            raise FileNotFoundError(f"Arquivo não encontrado: {mortality_path}")
    except Exception as e:
        st.warning(f"FERTILIDADE: Falha ao carregar de '{fertility_path}' ({e}). Usando taxas placeholder (TFR ~2.15 - ACIMA da reposição).") # Aviso modificado
        # --- Lógica Placeholder MELHORADA e COM FERTILIDADE MAIS ALTA ---
        fertility = {c: 0.0 for c in cohorts}
        step = params["demographics"]["cohort_step"]
        # <<< MUDANÇA PRINCIPAL: Aumenta o TFR alvo para acima da reposição >>>
        target_tfr = 2.15 # Ex: Ligeiramente acima da reposição para gerar crescimento

        # Perfil ASFR mais realista (pico 30-34) - taxas anuais por mulher
        # (Mantendo o mesmo perfil de distribuição etária)
        asfr_profile_shape = {
            "15-19": 0.007, "20-24": 0.048, "25-29": 0.090, "30-34": 0.100,
            "35-39": 0.060, "40-44": 0.015, "45-49": 0.001
        }
        implied_tfr_shape = sum(rate * step for cohort, rate in asfr_profile_shape.items())
        # Escala o perfil para atingir o NOVO target_tfr
        scaling_factor = target_tfr / implied_tfr_shape if implied_tfr_shape > 0 else 0
        scaled_asfr = {cohort: rate * scaling_factor for cohort, rate in asfr_profile_shape.items()}

        # Preenche o dicionário base do ano
        base_fertility_year = {c: 0.0 for c in cohorts}
        for cohort, rate in scaled_asfr.items():
            if cohort in base_fertility_year: base_fertility_year[cohort] = rate

        # Aplica pequena redução anual (opcional, pode ser 1.0 para manter estável)
        fertility_reduction_factor_annual = 1.0 # <<< MUDANÇA: Sem redução anual para simplificar
        current_fertility = base_fertility_year.copy()
        for year in years_needed:
             fertility_rates_proj[year] = current_fertility.copy()
             # Aplica o fator de redução apenas se for diferente de 1.0
             if fertility_reduction_factor_annual != 1.0:
                  current_fertility = {c: max(0.0001, rate * fertility_reduction_factor_annual) if c in scaled_asfr else 0.0 for c, rate in current_fertility.items()}
    # Fim da seção Placeholder de Fertilidade
    
    # --- Tentar Carregar Fertilidade ---
    try:
        if os.path.exists(fertility_path):
            df_fert = pd.read_csv(fertility_path)
            # Validação básica
            if not {'year', 'cohort', 'fertility_rate_annual_per_woman'}.issubset(df_fert.columns):
                 raise ValueError("Colunas esperadas ausentes no CSV de fertilidade.")

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
                      if year > min(fertility_rates_proj_loaded.keys()):
                           last_valid_year_fert = max(y for y in fertility_rates_proj_loaded if y < year)
                           fertility_rates_proj[year] = fertility_rates_proj[last_valid_year_fert] # Reutiliza taxas
                      else: # Se anos iniciais faltarem, não há o que fazer senão erro ou placeholder
                           raise FileNotFoundError(f"Ano inicial {year} ausente em {fertility_path}")

            print(f"INFO: Dados de fertilidade carregados de: {fertility_path}") # Confirmação (opcional)
            data_loaded_successfully['fertility'] = True
        else:
            raise FileNotFoundError(f"Arquivo não encontrado: {fertility_path}")

    except Exception as e:
        st.warning(f"FERTILIDADE: Falha ao carregar de '{fertility_path}' ({e}). Usando taxas placeholder.")
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
        fertility_reduction_factor_annual = 0.998
        current_fertility = fertility.copy()
        for year in years_needed:
             fertility_rates_proj[year] = current_fertility.copy()
             current_fertility = {c: max(0.0001, rate * fertility_reduction_factor_annual) for c, rate in current_fertility.items()}

    # Remove o aviso genérico se AMBOS foram carregados com sucesso
    if data_loaded_successfully['mortality'] and data_loaded_successfully['fertility']:
        # Se chegou aqui sem exceção para ambos, podemos remover o aviso genérico se houver
        pass # Não precisa mais do st.warning("DEMOG RATES: Usando taxas placeholder.")
    elif not data_loaded_successfully['mortality'] or not data_loaded_successfully['fertility']:
         st.warning("DEMOG RATES: Usando taxas placeholder para Mortalidade e/ou Fertilidade devido a falha no carregamento.")

    return mortality_rates_proj, fertility_rates_proj

def project_population_detailed(prev_pop_dist, mortality_rates_yr, fertility_rates_yr, migration_net_total, params):
    """
    Projeta população com envelhecimento, mortes, nascimentos e migração (Lógica Revisada V3 - Final Cohort Fix).
    Assume que mortality_rates e fertility_rates são taxas ANUAIS.
    """
    cohorts = get_cohorts(params)
    step = params["demographics"]["cohort_step"] # Geralmente 5
    new_pop_dist = {c: 0.0 for c in cohorts} # Começa zerado
    total_prev_pop = sum(prev_pop_dist.values())

    # 1. Nascimentos e Sobrevivência -> População Final 0-4
    total_births = 0
    f_start_age = params["demographics"]["fertility_age_start"]
    f_end_age = params["demographics"]["fertility_age_end"]
    # (Lógica de cálculo de total_births - igual à V2 corrigida)
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
    # Aplica sobrevivência aos nascidos
    mortality_rate_0_4 = mortality_rates_yr.get(cohorts[0], 0.01)
    mortality_rate_0_4 = max(0, min(1, mortality_rate_0_4))
    survival_rate_step_0_4 = (1 - mortality_rate_0_4) ** step
    # Define a população final da primeira coorte
    new_pop_dist[cohorts[0]] = total_births * survival_rate_step_0_4

    # 2. Envelhecimento e Mortes para Coortes 1 em diante
    for i in range(1, len(cohorts)): # Começa da segunda coorte (índice 1)
        current_cohort = cohorts[i]
        prev_cohort = cohorts[i-1]

        pop_start_prev_cohort = prev_pop_dist.get(prev_cohort, 0)
        mortality_rate_prev_cohort = mortality_rates_yr.get(prev_cohort, 0.01)
        mortality_rate_prev_cohort = max(0, min(1, mortality_rate_prev_cohort))

        # Sobreviventes da coorte ANTERIOR que envelhecem para a coorte ATUAL
        survival_rate_step_prev = (1 - mortality_rate_prev_cohort) ** step
        pop_aging_into_current = pop_start_prev_cohort * survival_rate_step_prev

        if current_cohort == cohorts[-1]: # É a última coorte (90+)?
            # População final = (Quem envelheceu da 85-89) + (Quem já estava na 90+ e sobreviveu)
            pop_start_last_cohort = prev_pop_dist.get(current_cohort, 0) # Pop 90+ anterior
            mortality_rate_last = mortality_rates_yr.get(current_cohort, 0.15)
            mortality_rate_last = max(0, min(1, mortality_rate_last))
            survival_rate_step_last = (1 - mortality_rate_last) ** step
            survivors_within_last = pop_start_last_cohort * survival_rate_step_last

            # Define a população final da última coorte
            new_pop_dist[current_cohort] = pop_aging_into_current + survivors_within_last
        else:
            # Para coortes intermediárias (5-9 até 85-89), a população final
            # é composta APENAS por quem envelheceu da coorte anterior.
            # Quem sobreviveu DENTRO desta coorte será calculado na próxima iteração do loop
            # quando ela se tornar a 'prev_cohort'.
            new_pop_dist[current_cohort] = pop_aging_into_current

    # 3. Migração (aplicada ao final, sobre a distribuição resultante)
    current_total_pop_before_migration = sum(new_pop_dist.values())
    if current_total_pop_before_migration > 0:
         for cohort in cohorts:
              proportion = new_pop_dist.get(cohort, 0) / current_total_pop_before_migration
              migration_cohort = migration_net_total * proportion
              new_pop_dist[cohort] = max(0, new_pop_dist.get(cohort, 0) + migration_cohort)
    elif migration_net_total > 0:
        migration_per_cohort = migration_net_total / len(cohorts)
        for cohort in cohorts: new_pop_dist[cohort] = max(0, new_pop_dist.get(cohort, 0) + migration_per_cohort)

    # Garante não negativo
    for cohort in cohorts: new_pop_dist[cohort] = max(0, new_pop_dist.get(cohort, 0))

    # Retorna nascimentos totais em Milhões e a nova distribuição populacional
    return new_pop_dist, total_births / 1e6

def load_initial_labor_participation(params):
     st.warning("PART RATE: Usando taxas placeholder.")
     cohorts = get_cohorts(params)
     rates = {c: 0.0 for c in cohorts}
     w_start, w_end = params["demographics"]["working_age_start"], params["demographics"]["working_age_end"]
     step = params["demographics"]["cohort_step"]
     for i in range(w_start, w_end + 1, step):
         cohort = f"{i}-{i+step-1}"
         if cohort in rates: rates[cohort] = params["labor_market"]["placeholder_participation_rate_working_age"]
     for i in range(params["demographics"]["retirement_age"], params["demographics"]["max_age"] + 1, step):
          if i == params["demographics"]["max_age"]: cohort = f"{i}+"
          else: cohort = f"{i}-{i+step-1}"
          if cohort in rates: rates[cohort] = params["labor_market"]["placeholder_participation_rate_old"]
     return rates

def calculate_labor_input_detailed(pop_dist, participation_rates, avg_hours, unemployment_rate, ubi_impact_participation, ubi_impact_hours, ubi_active, params):
    total_hours = 0
    w_start, w_end = params["demographics"]["working_age_start"], params["demographics"]["working_age_end"]
    retire_age = params["demographics"]["retirement_age"]
    for cohort, pop in pop_dist.items():
        try: age_min = int(cohort.split('-')[0])
        except: age_min = params["demographics"]["max_age"]
        base_rate = participation_rates.get(cohort, 0)
        ubi_effect_key = None
        if w_start <= age_min <= w_end: ubi_effect_key = "working_age"
        elif age_min >= retire_age: ubi_effect_key = "old_age"
        rate_reduction = ubi_impact_participation.get(ubi_effect_key, 0) if ubi_active and ubi_effect_key else 0
        hours_reduction = ubi_impact_hours if ubi_active and ubi_effect_key == "working_age" else 0
        current_part_rate = base_rate * (1 - rate_reduction)
        employed = pop * current_part_rate * (1 - unemployment_rate)
        current_hours = avg_hours * (1 - hours_reduction)
        total_hours += employed * current_hours * 1_000_000
    return total_hours

def load_initial_economy(params):
    """Placeholder: Carrega dados econômicos iniciais."""
    st.warning("ECON: Usando dados iniciais placeholder. Carregar de: " + params['data_paths']['initial_economy'])
    # TODO: Implementar leitura de params["data_paths"]["initial_economy"] (CSV/API)
    # que deve conter colunas/valores para GDP, Capital, TFP, PriceLevel Iniciais.

    # Placeholder return value (valores fixos para exemplo):
    # Estes valores deveriam vir do arquivo lido.
    initial_gdp = 4.0
    initial_capital_ratio = 2.8 # Este valor pode vir do arquivo ou ser calculado/calibrado
    initial_capital = initial_gdp * initial_capital_ratio
    initial_tfp = 100.0
    initial_price_level = 100.0

    # Você pode adicionar outros valores iniciais necessários aqui, lidos do arquivo
    # initial_avg_hours = ...
    # initial_unemployment = ...

    return {
        "GDP": initial_gdp,
        "Capital": initial_capital,
        "TFP": initial_tfp,
        "PriceLevel": initial_price_level
        # Adicione outros valores lidos aqui se necessário para o _initialize_scenario_dataframe
        # "Avg_Hours_Worked": initial_avg_hours,
        # "Unemployment_Rate": initial_unemployment,
        # ...
    }

def calibrate_economy(params):
     st.warning("CALIB: Calibração não implementada. Usando placeholders.")
     return params["economy"]["cobb_douglas_alpha"], params["economy"]["tfp_calibration_factor"]

def calculate_capital_stock(prev_capital, investment_trillions, depreciation_rate):
    return prev_capital * (1 - depreciation_rate) + investment_trillions

def calculate_gdp_cobb_douglas(tfp_level, capital_stock, labor_input_hours, alpha, scale_factor):
    if capital_stock <= 0 or labor_input_hours <= 0 or tfp_level <= 0: return 0
    # scale_factor deve ser calibrado offline para TFP=100 e K, L iniciais resultarem em Y inicial
    gdp = scale_factor * tfp_level * (capital_stock**alpha) * (labor_input_hours**(1-alpha))
    return gdp

def calculate_tfp(prev_tfp, base_growth, ubi_boost, edu_effect):
     return prev_tfp * (1 + base_growth + ubi_boost) # Ignora edu_effect

def calculate_investment(gdp, rate):
     return gdp * rate

def load_initial_social(params):
    st.warning("SOCIAL: Usando dados iniciais placeholder.")
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
         new_indicators["Poverty"] = max(poverty_floor, poverty_reduced)
    return new_indicators

def calculate_ubi_cost_detailed(pop_dist, ubi_amount_nominal, params):
    cost = 0
    min_age = params["ubi"]["min_eligible_age"]
    # TODO: Carência imigrantes
    for cohort, pop in pop_dist.items():
         try: age_min = int(cohort.split('-')[0])
         except: age_min = params["demographics"]["max_age"]
         if age_min >= min_age: cost += pop * 1e6 * ubi_amount_nominal
    return cost / 1e9 # Bilhões EUR

def calculate_fiscal_balance(gdp_real, price_level_index, gdp_real_prev, price_level_prev, ubi_cost_b, prev_debt_ratio, params):
     """Calcula balanço fiscal e nova dívida/PIB (nominal)."""
     gdp_nominal_t = gdp_real * (price_level_index / 100) if price_level_index else gdp_real # PIB Nominal Atual
     gdp_nominal_prev = gdp_real_prev * (price_level_prev / 100) if price_level_prev else gdp_real_prev # PIB Nominal Anterior

     # ... (cálculo de other_spending_b, baseline_revenue_b, additional_tax_b, ubi_deficit_b como antes) ...
     other_spending_b = (params["government_finance"]["other_gov_spending_share_gdp"] * gdp_nominal_t) * 1000 if gdp_nominal_t > 0 else 0
     baseline_revenue_b = (params["government_finance"]["baseline_tax_revenue_share_gdp"] * gdp_nominal_t) * 1000 if gdp_nominal_t > 0 else 0
     ubi_tax_coverage = params["government_finance"]["ubi_financing_tax_share"]
     additional_tax_b = ubi_cost_b * ubi_tax_coverage
     ubi_deficit_b = ubi_cost_b * (1 - ubi_tax_coverage)

     total_spending_b = other_spending_b + ubi_cost_b
     total_revenue_b = baseline_revenue_b + additional_tax_b

     deficit_b = total_spending_b - total_revenue_b
     deficit_gdp_ratio = (deficit_b / 1000) / gdp_nominal_t if gdp_nominal_t > 0 else 0

     # Atualização Dívida/PIB (Nominal) - CORRIGIDO
     # Calcula a dívida nominal anterior a partir do ratio e PIB nominal anterior
     prev_debt_nominal = prev_debt_ratio * gdp_nominal_prev if gdp_nominal_prev is not None else 0 # Dívida Nominal Anterior em Trilhões

     current_debt_nominal = prev_debt_nominal + (deficit_b / 1000) # Nova Dívida Nominal em Trilhões
     new_debt_ratio = current_debt_nominal / gdp_nominal_t if gdp_nominal_t > 0 else prev_debt_ratio # Dívida Nominal / PIB Nominal

     return deficit_gdp_ratio, new_debt_ratio


# --- Novas Funções para PIB Potencial e Inflação ---
def calculate_potential_gdp(tfp_level, capital_stock, pop_dist, participation_rates, avg_hours, nairu, alpha, scale_factor, params):
    """Calcula o PIB Potencial usando NAIRU."""
    # Calcula input de trabalho potencial (usando NAIRU)
    potential_labor_input = 0
    w_start, w_end = params["demographics"]["working_age_start"], params["demographics"]["working_age_end"]
    retire_age = params["demographics"]["retirement_age"]
    for cohort, pop in pop_dist.items():
         # Usa taxas de participação BASELINE (ou uma estimativa de longo prazo)
         # Aqui, simplificamos usando as taxas carregadas inicialmente
         base_rate = params["labor_market"]["placeholder_participation_rate_working_age"] # Simplificação grosseira!! Deveria usar taxas por coorte base
         try: age_min = int(cohort.split('-')[0])
         except: age_min = params["demographics"]["max_age"]
         if not (w_start <= age_min <= w_end or age_min >= retire_age): continue # Apenas idades ativas/idosas

         employed_potential = pop * base_rate * (1 - nairu) # Emprego com NAIRU
         potential_labor_input += employed_potential * avg_hours * 1_000_000

    if potential_labor_input <= 0: return 0

    potential_gdp = calculate_gdp_cobb_douglas(tfp_level * scale_factor, capital_stock, potential_labor_input, alpha, scale_factor)
    return potential_gdp

def calculate_inflation(baseline_inflation, gdp_real, gdp_potential, ubi_cost_b, financing_params, econ_params):
    """Calcula a taxa de inflação (simplificado)."""
    if gdp_potential <= 0: return baseline_inflation

    output_gap = (gdp_real - gdp_potential) / gdp_potential

    # Estímulo de demanda da RBU (% do PIB Potencial)
    prop_consume = financing_params["consumption_propensity_ubi_recipients"]
    tax_share = financing_params["ubi_financing_tax_share"]
    prop_reduce_consume_tax = financing_params["consumption_reduction_high_income_financing"]
    # Demanda bruta da RBU * propensão consumir - Demanda reduzida pelos impostos que financiam
    net_demand_stimulus_b = (ubi_cost_b * prop_consume) - (ubi_cost_b * tax_share * prop_reduce_consume_tax)
    demand_stimulus_ratio = (net_demand_stimulus_b / 1000) / gdp_potential if gdp_potential > 0 else 0

    # Regra de Inflação
    gap_sensitivity = econ_params["inflation_output_gap_sensitivity"]
    demand_sensitivity = econ_params["inflation_ubi_demand_sensitivity"]

    # A inflação aumenta com hiato positivo (demanda > oferta) e com estímulo da UBI
    # O efeito é maior se o hiato já for pequeno/positivo
    gap_effect = gap_sensitivity * max(0, output_gap) # Inflação só reage a hiato positivo? Ou simétrico? Simplificado: só positivo.
    # O efeito do estímulo UBI pode ser maior se a economia já está aquecida (hiato pequeno)
    ubi_effect = demand_sensitivity * demand_stimulus_ratio * (1 / (1 + max(0, -output_gap*2))) # Amplifica se output gap for pequeno/negativo

    inflation = baseline_inflation + gap_effect + ubi_effect

    # Aplica teto para evitar divergência
    inflation = min(inflation, econ_params.get("inflation_ceiling", 0.10))
    return max(0, inflation) # Garante não negativa


# --- Classe Simulador Principal (Atualizada) ---
class UBISimulatorRealisticV3:
    def __init__(self, scenario_params):
        self.params = scenario_params
        sim_years = self.params["simulation_years"]
        self.start_year = sim_years["start"]
        self.end_year = sim_years["end"]
        self.years = np.arange(self.start_year, self.end_year + 1)
        self.results = {}
        self.cohorts = get_cohorts(self.params)

        # Carregar e calibrar (placeholders)
        self.initial_pop_dist = load_initial_population(self.params)
        self.mortality_rates, self.fertility_rates = load_demographic_rates(self.params)
        self.initial_participation = load_initial_labor_participation(self.params) # Usado para Potencial
        self.initial_economy = load_initial_economy(self.params)
        self.initial_social = load_initial_social(self.params)
        self.alpha, self.tfp_factor = calibrate_economy(self.params) # tfp_factor é o 'A' inicial escalado

    def _initialize_scenario_dataframe(self):
        columns = [
            "Total Population (M)", "GDP Real (Trillion EUR)", "GDP Growth Rate",
            "GDP Potential (Trillion EUR)", "Output Gap (%)", # Novo
            "Inflation Rate", "Price Level Index", # Novo
            "Capital Stock (Trillion EUR)", "Investment (Trillion EUR)",
            "TFP Level (Index)", "TFP Growth Rate",
            "Labor Input (Billion Hours)", "Unemployment Rate", "Avg Hours Worked",
            "UBI Cost (Billion EUR)", "UBI Nominal Amount (€)", # Novo
            "Govt Deficit/GDP Ratio", "Govt Debt/GDP Ratio",
            "Gini Coefficient", "Poverty Rate", "Births (Millions)"
        ]
        for cohort in self.cohorts: columns.extend([f"Pop {cohort} (M)", f"Part Rate {cohort}"])
        df = pd.DataFrame(index=self.years, columns=columns, dtype=float)
        base_year = self.start_year - 1
        # Preenche ano base... (similar ao anterior, adicionando Price Level)
        df.loc[base_year, "Total Population (M)"] = sum(self.initial_pop_dist.values())
        for cohort, pop in self.initial_pop_dist.items(): df.loc[base_year, f"Pop {cohort} (M)"] = pop
        for cohort, rate in self.initial_participation.items(): df.loc[base_year, f"Part Rate {cohort}"] = rate
        df.loc[base_year, "GDP Real (Trillion EUR)"] = self.initial_economy["GDP"]
        df.loc[base_year, "Capital Stock (Trillion EUR)"] = self.initial_economy["Capital"]
        df.loc[base_year, "TFP Level (Index)"] = self.initial_economy["TFP"]
        df.loc[base_year, "Price Level Index"] = self.initial_economy["PriceLevel"] # Inicial = 100
        df.loc[base_year, "Inflation Rate"] = self.params["economy"]["baseline_inflation_rate"] # Assume inflação base no ano anterior
        df.loc[base_year, "Unemployment Rate"] = self.params["labor_market"]["baseline_unemployment_rate"]
        df.loc[base_year, "Avg Hours Worked"] = self.params["labor_market"]["placeholder_avg_hours"]
        df.loc[base_year, "Gini Coefficient"] = self.initial_social["Gini"]
        df.loc[base_year, "Poverty Rate"] = self.initial_social["Poverty"]
        df.loc[base_year, "Govt Debt/GDP Ratio"] = self.params["government_finance"]["initial_debt_gdp_ratio"]
        df.loc[base_year, "Births (Millions)"] = 0
        df.loc[base_year, "UBI Cost (Billion EUR)"] = 0
        df.loc[base_year, "UBI Nominal Amount (€)"] = self.params["ubi"]["annual_amount_eur_real_start"] # Começa com valor real
        df.loc[base_year, "Govt Deficit/GDP Ratio"] = 0
        # Calcular GDP Potencial inicial
        gdp_pot_init = calculate_potential_gdp(df.loc[base_year, "TFP Level (Index)"], df.loc[base_year, "Capital Stock (Trillion EUR)"],
                                             self.initial_pop_dist, self.initial_participation, df.loc[base_year, "Avg Hours Worked"],
                                             self.params["labor_market"]["nairu"], self.alpha, self.tfp_factor, self.params)
        df.loc[base_year, "GDP Potential (Trillion EUR)"] = gdp_pot_init
        output_gap_init = (self.initial_economy["GDP"] - gdp_pot_init) / gdp_pot_init if gdp_pot_init else 0
        df.loc[base_year, "Output Gap (%)"] = output_gap_init * 100
        return df # Não preenche NaNs ainda


    def run_simulation(self):
        df_baseline = self._initialize_scenario_dataframe()
        df_ubi = self._initialize_scenario_dataframe()

        # --- Loop Anual ---
        for year in self.years:
            prev_year = year - 1

            # --- Simulação Baseline ---
            # (Demografia, Trabalho, TFP, Capital, GDP Real - igual V2)

            # ... (código da V2 para baseline) ...
            prev_pop_dist_base = {c: df_baseline.loc[prev_year, f"Pop {c} (M)"] for c in self.cohorts}
            migration_total_base = sum(prev_pop_dist_base.values()) * self.params["demographics"]["placeholder_net_migration_share_of_pop"]
            current_pop_dist_base, births_base = project_population_detailed(prev_pop_dist_base, self.mortality_rates, self.fertility_rates, migration_total_base, self.params)
            df_baseline.loc[year, "Births (Millions)"] = births_base / 1e6 # Salva em milhões
            for cohort, pop in current_pop_dist_base.items(): df_baseline.loc[year, f"Pop {cohort} (M)"] = pop
            df_baseline.loc[year, "Total Population (M)"] = sum(current_pop_dist_base.values())
            part_rates_base = {c: df_baseline.loc[prev_year, f"Part Rate {c}"] for c in self.cohorts}
            for c, r in part_rates_base.items(): df_baseline.loc[year, f"Part Rate {c}"] = r
            df_baseline.loc[year, "Unemployment Rate"] = df_baseline.loc[prev_year, "Unemployment Rate"]
            df_baseline.loc[year, "Avg Hours Worked"] = df_baseline.loc[prev_year, "Avg Hours Worked"]
            labor_input_base = calculate_labor_input_detailed(current_pop_dist_base, part_rates_base, df_baseline.loc[year, "Avg Hours Worked"], df_baseline.loc[year, "Unemployment Rate"], {}, 0, False, self.params)
            df_baseline.loc[year, "Labor Input (Billion Hours)"] = labor_input_base / 1e9
            tfp_base = calculate_tfp(df_baseline.loc[prev_year, "TFP Level (Index)"], self.params["economy"]["baseline_tfp_growth_rate"], 0, 0)
            df_baseline.loc[year, "TFP Level (Index)"] = tfp_base
            df_baseline.loc[year, "TFP Growth Rate"] = tfp_base / df_baseline.loc[prev_year, "TFP Level (Index)"] - 1 if df_baseline.loc[prev_year, "TFP Level (Index)"] else 0
            gdp_prev_base = df_baseline.loc[prev_year, "GDP Real (Trillion EUR)"]
            investment_base = calculate_investment(gdp_prev_base, self.params["economy"]["investment_rate_gdp"])
            df_baseline.loc[year, "Investment (Trillion EUR)"] = investment_base
            capital_stock_base = calculate_capital_stock(df_baseline.loc[prev_year, "Capital Stock (Trillion EUR)"], investment_base, self.params["economy"]["capital_depreciation_rate"])
            df_baseline.loc[year, "Capital Stock (Trillion EUR)"] = capital_stock_base
            gdp_base = calculate_gdp_cobb_douglas(tfp_base, capital_stock_base, labor_input_base, self.alpha, self.tfp_factor)
            df_baseline.loc[year, "GDP Real (Trillion EUR)"] = gdp_base
            df_baseline.loc[year, "GDP Growth Rate"] = (gdp_base / gdp_prev_base - 1) if gdp_prev_base else 0

            # Inflação e Potencial (Baseline)
            gdp_potential_base = calculate_potential_gdp(tfp_base, capital_stock_base, current_pop_dist_base, self.initial_participation, # Usa part. inicial para potencial
                                                         df_baseline.loc[year, "Avg Hours Worked"], self.params["labor_market"]["nairu"],
                                                         self.alpha, self.tfp_factor, self.params)
            df_baseline.loc[year, "GDP Potential (Trillion EUR)"] = gdp_potential_base
            output_gap_base = (gdp_base - gdp_potential_base) / gdp_potential_base if gdp_potential_base else 0
            df_baseline.loc[year, "Output Gap (%)"] = output_gap_base * 100
            # Inflação baseline = inflação base (sem estímulo UBI) + efeito hiato
            inflation_base = self.params["economy"]["baseline_inflation_rate"] + \
                             self.params["economy"]["inflation_output_gap_sensitivity"] * max(0, output_gap_base)
            inflation_base = min(inflation_base, self.params["economy"].get("inflation_ceiling", 0.10))
            inflation_base = max(0, inflation_base)
            df_baseline.loc[year, "Inflation Rate"] = inflation_base
            df_baseline.loc[year, "Price Level Index"] = df_baseline.loc[prev_year, "Price Level Index"] * (1 + inflation_base)

            # Fiscal (Baseline)
            df_baseline.loc[year, "UBI Cost (Billion EUR)"] = 0
            df_baseline.loc[year, "UBI Nominal Amount (€)"] = 0 # Sem UBI
            deficit_ratio_base, debt_ratio_base = calculate_fiscal_balance(
                gdp_base,
                df_baseline.loc[year, "Price Level Index"],
                df_baseline.loc[prev_year, "GDP Real (Trillion EUR)"], # gdp_real_prev
                df_baseline.loc[prev_year, "Price Level Index"],      # price_level_prev
                0, # ubi_cost_b
                df_baseline.loc[prev_year, "Govt Debt/GDP Ratio"],
                self.params
            )
            df_baseline.loc[year, "Govt Deficit/GDP Ratio"] = deficit_ratio_base
            df_baseline.loc[year, "Govt Debt/GDP Ratio"] = debt_ratio_base

            # Social (Baseline)
            # ... (código V2) ...
            prev_social_base = {"Gini": df_baseline.loc[prev_year, "Gini Coefficient"], "Poverty": df_baseline.loc[prev_year, "Poverty Rate"]}
            current_social_base = update_social_indicators(prev_social_base, False, self.params) # False = ubi_active
            df_baseline.loc[year, "Gini Coefficient"] = current_social_base["Gini"]
            df_baseline.loc[year, "Poverty Rate"] = current_social_base["Poverty"]


            # --- Simulação Com RBU ---
            ubi_active = year >= self.params["ubi"]["start_year"]

            # Ajusta taxas de fertilidade se UBI ativa
            current_fertility_rates = self.fertility_rates.get(year, {}) # Pega taxas do ano
            boost = self.params["ubi"].get("fertility_rate_boost_factor", 0.0) # Pega o fator (default 0)

            # # DEBUG: Imprimir a variável antes do erro
            # print(f"\n--- DEBUG: Ano {year} para fertility_rates_ubi ---")
            # print(f"Tipo de current_fertility_rates: {type(current_fertility_rates)}")
            # # Imprime alguns itens para ver a estrutura dos valores (rate)
            # items_sample = list(current_fertility_rates.items())[:5] # Pega os 5 primeiros itens
            # print(f"Amostra de itens (cohort, rate): {items_sample}")
            # print("Verificando tipo de 'rate' em cada item da amostra:")
            # for i, (c, r) in enumerate(items_sample):
            #     print(f"  Item {i}: cohort='{c}', type(rate)={type(r)}, rate value={r}")
            #     # Tentativa de acesso se for dicionário (para diagnóstico)
            #     if isinstance(r, dict):
            #         print(f"    (rate é dict, chaves: {list(r.keys())})")
            # print(f"Valor de boost: {boost} (tipo: {type(boost)})")
            # print(f"--- FIM DEBUG ---\n")

            # Linha original que causa o erro:
            try:
                fertility_rates_ubi = {cohort: rate * (1 + boost) for cohort, rate in current_fertility_rates.items()}
            except TypeError as e:
                print(f"!!! ERRO NA COMPREENSÃO: {e}")
                print("!!! Verifique a estrutura de 'rate' acima.")
                # Para o loop ou levanta o erro para interromper a execução
                raise e # Re-levanta o erro após o debug print


            # Demografia (passa as taxas ajustadas ou não)
            prev_pop_dist_ubi = {c: df_ubi.loc[prev_year, f"Pop {c} (M)"] for c in self.cohorts}
            migration_total_ubi = sum(prev_pop_dist_ubi.values()) * self.params["demographics"]["placeholder_net_migration_share_of_pop"] # TODO: Modelo Migração
            current_pop_dist_ubi, births_ubi = project_population_detailed(
                prev_pop_dist_ubi,
                self.mortality_rates,
                fertility_rates_ubi, # <--- Usa as taxas potencialmente ajustadas
                migration_total_ubi,
                self.params
            )

            #
            df_ubi.loc[year, "Births (Millions)"] = births_ubi / 1e6
            for cohort, pop in current_pop_dist_ubi.items(): df_ubi.loc[year, f"Pop {cohort} (M)"] = pop
            df_ubi.loc[year, "Total Population (M)"] = sum(current_pop_dist_ubi.values())

            # UBI Nominal (Indexado)
            prev_inflation_ubi = df_ubi.loc[prev_year, "Inflation Rate"]
            prev_ubi_nominal = df_ubi.loc[prev_year, "UBI Nominal Amount (€)"]
            current_ubi_nominal = 0
            if ubi_active:
                 # Se for o primeiro ano, usa o valor real inicial como nominal
                 if year == self.params["ubi"]["start_year"]:
                      current_ubi_nominal = self.params["ubi"]["annual_amount_eur_real_start"]
                 else:
                      # Indexa pelo lag definido (normalmente ano anterior)
                      indexation_lag = self.params["ubi"].get("indexation_lag", 1)
                      inflation_to_index = df_ubi.loc[year - indexation_lag, "Inflation Rate"] if (year - indexation_lag) >= self.start_year -1 else self.params["economy"]["baseline_inflation_rate"]
                      current_ubi_nominal = prev_ubi_nominal * (1 + inflation_to_index)
            df_ubi.loc[year, "UBI Nominal Amount (€)"] = current_ubi_nominal

            # Trabalho (Impactado pela UBI)
            # ... (código V2) ...
            part_rates_ubi = {}
            for c in self.cohorts:
                 base_rate = df_ubi.loc[prev_year, f"Part Rate {c}"]
                 reduction_factor = self.params["labor_market"]["ubi_participation_reduction_factor"]
                 ubi_effect_key = None
                 try: age_min = int(c.split('-')[0])
                 except: age_min = self.params["demographics"]["max_age"]
                 if self.params["demographics"]["working_age_start"] <= age_min <= self.params["demographics"]["working_age_end"]: ubi_effect_key = "working_age"
                 elif age_min >= self.params["demographics"]["retirement_age"]: ubi_effect_key = "old_age"
                 reduction = reduction_factor.get(ubi_effect_key, 0) if ubi_active and ubi_effect_key else 0
                 part_rates_ubi[c] = base_rate * (1 - reduction)
                 df_ubi.loc[year, f"Part Rate {c}"] = part_rates_ubi[c]
            df_ubi.loc[year, "Unemployment Rate"] = df_ubi.loc[prev_year, "Unemployment Rate"]
            hours_reduction_ubi = self.params["labor_market"]["ubi_hours_reduction_factor"] if ubi_active else 0
            df_ubi.loc[year, "Avg Hours Worked"] = df_ubi.loc[prev_year, "Avg Hours Worked"] * (1 - hours_reduction_ubi)
            labor_input_ubi = calculate_labor_input_detailed(current_pop_dist_ubi, part_rates_ubi, df_ubi.loc[year, "Avg Hours Worked"], df_ubi.loc[year, "Unemployment Rate"], self.params["labor_market"]["ubi_participation_reduction_factor"], self.params["labor_market"]["ubi_hours_reduction_factor"], ubi_active, self.params)
            df_ubi.loc[year, "Labor Input (Billion Hours)"] = labor_input_ubi / 1e9


            # Economia (com TFP boost)
            # ... (código V2) ...
            ubi_tfp_boost = self.params["economy"]["ubi_tfp_growth_boost_pp"] if ubi_active else 0
            tfp_ubi = calculate_tfp(df_ubi.loc[prev_year, "TFP Level (Index)"], self.params["economy"]["baseline_tfp_growth_rate"], ubi_tfp_boost, 0)
            df_ubi.loc[year, "TFP Level (Index)"] = tfp_ubi
            df_ubi.loc[year, "TFP Growth Rate"] = tfp_ubi / df_ubi.loc[prev_year, "TFP Level (Index)"] - 1 if df_ubi.loc[prev_year, "TFP Level (Index)"] else 0
            gdp_prev_ubi = df_ubi.loc[prev_year, "GDP Real (Trillion EUR)"]
            investment_ubi = calculate_investment(gdp_prev_ubi, self.params["economy"]["investment_rate_gdp"])
            df_ubi.loc[year, "Investment (Trillion EUR)"] = investment_ubi
            capital_stock_ubi = calculate_capital_stock(df_ubi.loc[prev_year, "Capital Stock (Trillion EUR)"], investment_ubi, self.params["economy"]["capital_depreciation_rate"])
            df_ubi.loc[year, "Capital Stock (Trillion EUR)"] = capital_stock_ubi
            gdp_ubi = calculate_gdp_cobb_douglas(tfp_ubi, capital_stock_ubi, labor_input_ubi, self.alpha, self.tfp_factor)
            df_ubi.loc[year, "GDP Real (Trillion EUR)"] = gdp_ubi
            df_ubi.loc[year, "GDP Growth Rate"] = (gdp_ubi / gdp_prev_ubi - 1) if gdp_prev_ubi else 0


            # Inflação e Potencial (Com UBI)
            gdp_potential_ubi = calculate_potential_gdp(tfp_ubi, capital_stock_ubi, current_pop_dist_ubi, self.initial_participation,
                                                        df_ubi.loc[year, "Avg Hours Worked"], self.params["labor_market"]["nairu"],
                                                        self.alpha, self.tfp_factor, self.params)
            df_ubi.loc[year, "GDP Potential (Trillion EUR)"] = gdp_potential_ubi
            output_gap_ubi = (gdp_ubi - gdp_potential_ubi) / gdp_potential_ubi if gdp_potential_ubi else 0
            df_ubi.loc[year, "Output Gap (%)"] = output_gap_ubi * 100

            # Custo da RBU (Nominal) - Calculado antes da inflação para usar no cálculo dela
            ubi_cost_b = calculate_ubi_cost_detailed(current_pop_dist_ubi, current_ubi_nominal, self.params) if ubi_active else 0
            df_ubi.loc[year, "UBI Cost (Billion EUR)"] = ubi_cost_b

            # Calcular Inflação
            inflation_ubi = calculate_inflation(
                self.params["economy"]["baseline_inflation_rate"],
                gdp_ubi, gdp_potential_ubi, ubi_cost_b,
                self.params["government_finance"], self.params["economy"]
            )
            df_ubi.loc[year, "Inflation Rate"] = inflation_ubi
            df_ubi.loc[year, "Price Level Index"] = df_ubi.loc[prev_year, "Price Level Index"] * (1 + inflation_ubi)

            # Fiscal (Com UBI)
            deficit_ratio_ubi, debt_ratio_ubi = calculate_fiscal_balance(
                gdp_ubi,
                df_ubi.loc[year,"Price Level Index"],
                df_ubi.loc[prev_year, "GDP Real (Trillion EUR)"], # gdp_real_prev
                df_ubi.loc[prev_year, "Price Level Index"],      # price_level_prev
                ubi_cost_b,
                df_ubi.loc[prev_year, "Govt Debt/GDP Ratio"],
                self.params
            )
            df_ubi.loc[year, "Govt Deficit/GDP Ratio"] = deficit_ratio_ubi
            df_ubi.loc[year, "Govt Debt/GDP Ratio"] = debt_ratio_ubi

            # Social (Com UBI)
            # ... (código V2) ...
            prev_social_ubi = {"Gini": df_ubi.loc[prev_year, "Gini Coefficient"], "Poverty": df_ubi.loc[prev_year, "Poverty Rate"]}
            current_social_ubi = update_social_indicators(prev_social_ubi, ubi_active, self.params)
            df_ubi.loc[year, "Gini Coefficient"] = current_social_ubi["Gini"]
            df_ubi.loc[year, "Poverty Rate"] = current_social_ubi["Poverty"]


        # Armazena resultados finais
        self.results["Baseline (Sem RBU)"] = df_baseline.drop(self.start_year - 1).fillna(0)
        self.results["Com RBU"] = df_ubi.drop(self.start_year - 1).fillna(0)

    # get_results e get_summary_dataframe (adaptados para novos indicadores)
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
scenario_params = DEFAULT_SCENARIO_PARAMS

# Controles UI
ubi_amount_real_start = st.sidebar.slider("Valor Anual REAL Inicial RBU (€)", 0, 25000, scenario_params["ubi"]["annual_amount_eur_real_start"], 500)
ubi_start_year = st.sidebar.slider("Ano Início RBU", scenario_params["simulation_years"]["start"], scenario_params["simulation_years"]["end"] - 1, scenario_params["ubi"]["start_year"])
labor_reduction_active = st.sidebar.slider("Redução Participação (Idade Ativa, %)", 0.0, 15.0, scenario_params["labor_market"]["ubi_participation_reduction_factor"]["working_age"] * 100, 0.5, "%.1f%%")
tfp_boost = st.sidebar.slider("Boost TFP Anual (p.p.)", 0.0, 0.5, scenario_params["economy"]["ubi_tfp_growth_boost_pp"] * 100, 0.05, "%.2f p.p.")
inflation_demand_sens = st.sidebar.slider("Sensibilidade Inflação->Demanda UBI", 0.0, 1.0, scenario_params["economy"]["inflation_ubi_demand_sensitivity"], 0.05)

# Atualiza parâmetros
current_params = scenario_params.copy()
current_params["ubi"]["annual_amount_eur_real_start"] = ubi_amount_real_start
current_params["ubi"]["start_year"] = ubi_start_year
current_params["labor_market"]["ubi_participation_reduction_factor"]["working_age"] = labor_reduction_active / 100
current_params["economy"]["ubi_tfp_growth_boost_pp"] = tfp_boost / 100
current_params["economy"]["inflation_ubi_demand_sensitivity"] = inflation_demand_sens


# --- Execução e Exibição ---
@st.cache_data
def run_simulation_cached_v3(params):
    simulator = UBISimulatorRealisticV3(params)
    simulator.run_simulation()
    return simulator

simulator = run_simulation_cached_v3(current_params)
results = simulator.get_results()

if not results:
    st.error("Falha ao executar simulação.")
else:
    st.header("Resultados")
    baseline_df = results["Baseline (Sem RBU)"]
    ubi_df = results["Com RBU"]

    # Tabela Resumo
    st.subheader("Resumo Anual")
    year_display = st.slider("Ano Tabela:", min_value=current_params["simulation_years"]["start"], max_value=current_params["simulation_years"]["end"], value=current_params["simulation_years"]["end"], key="table_year_v3")
    summary = simulator.get_summary_dataframe(year_display)
    if not summary.empty: st.dataframe(summary, use_container_width=True)
    else: st.warning(f"Não foi possível gerar a tabela para o ano {year_display}.")

    # Gráficos Principais
    st.subheader("Evolução Temporal")
    charts_v3 = ["GDP Real (Trillion EUR)", "Inflation Rate", "Price Level Index", "UBI Nominal Amount (€)", "Govt Debt/GDP Ratio", "Gini Coefficient", "Poverty Rate", "UBI Cost (Billion EUR)"]
    for chart_ind in charts_v3:
         if chart_ind in baseline_df.columns and chart_ind in ubi_df.columns:
             st.markdown(f"**{chart_ind}**")
             # Trata 'UBI Nominal Amount (€)' que é 0 no baseline
             if chart_ind == "UBI Nominal Amount (€)":
                  chart_data = pd.DataFrame({'Com RBU': ubi_df[chart_ind]})
             else:
                  chart_data = pd.DataFrame({'Sem RBU': baseline_df[chart_ind], 'Com RBU': ubi_df[chart_ind]})
             st.line_chart(chart_data)
         else:
             st.warning(f"Indicador '{chart_ind}' não plotado.")

    # Detalhes Demográficos
    with st.expander("Detalhes Demográficos (Coortes)"):
         # ... (código do expander anterior) ...
         pop_cols = [f"Pop {c} (M)" for c in simulator.cohorts]
         pop_df_base = baseline_df[pop_cols]
         pop_df_ubi = ubi_df[pop_cols]
         st.markdown("**População por Coorte - Cenário Base**"); st.line_chart(pop_df_base)
         st.markdown("**População por Coorte - Cenário com RBU**"); st.line_chart(pop_df_ubi)