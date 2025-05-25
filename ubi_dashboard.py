import streamlit as st
import pandas as pd
import numpy as np
import datetime # Para obter o ano atual como sugestão
from ubi import UBISimulatorRealisticV3, DEFAULT_SCENARIO_PARAMS # Import new simulator

# --- Default Parameters (Ajustados e com Fertilidade) ---
# Valores internos representam a escala usada nos cálculos (ex: 0.04 para 4%)
# DEFAULT_PARAMS = {
#     # Parâmetros da RBU
#     "ubi_annual_amount_eur": 14400.0, # Usar float para consistência
#     "ubi_start_year": 2027,           # Inteiro
#     "ubi_eligible_population_share": 1.0,

#     # Parâmetros Econômicos e Comportamentais
#     "labor_participation_reduction_factor": 0.04,
#     "labor_hours_reduction_factor": 0.01,
#     "productivity_growth_boost_pp": 0.001,
#     "consumption_propensity_ubi_recipients": 0.8,
#     "consumption_reduction_high_income_financing": 0.2,

#     # Parâmetros Sociais
#     "wellbeing_satisfaction_boost": 0.5,
#     "mental_health_reduction_pct": 40.0,
#     "crime_reduction_per_gini_point": 0.05,
#     "unpaid_work_increase_factor": 1.1,

#     # Parâmetros de Financiamento e Reação
#     "financing_model": "mixed_taxes_debt",
#     "additional_tax_rate_for_ubi": 0.05,
#     "capital_flight_factor_high_tax": 0.005,
#     "migration_net_increase_factor_restricted": 1.25,

#     # Parâmetros de Inflação
#     "baseline_annual_inflation_rate": 0.02,
#     "inflation_ubi_demand_sensitivity": 0.15,
#     "inflation_productivity_dampening_factor": 0.5,

#     # Impacto Populacional Adicional
#     "fertility_rate_ubi_boost_pp": 0.001,

#     # Parâmetros de Simulação - Default para inputs
#     "start_year": datetime.date.today().year, # Começa no ano atual por padrão
#     "end_year": datetime.date.today().year + 50, # Simula 50 anos por padrão

#     # Dados Iniciais
#     "initial_population_millions": 84.0,
#     "initial_gdp_trillion_eur": 4.0,
#     "initial_participation_rate": 0.75,
#     "initial_avg_hours_worked_per_worker": 1600.0, # Float
#     "initial_unemployment_rate": 0.055,
#     "initial_gini_coefficient": 0.29,
#     "initial_poverty_rate": 0.15,
#     "initial_wellbeing_satisfaction": 7.0,
#     "initial_mental_health_prevalence": 0.10,
#     "initial_crime_rate_index": 100.0,
#     "initial_net_migration_thousands": 100.0,
#     "initial_millionaires_thousands": 150.0,
#     "initial_govt_debt_gdp_ratio": 0.60,
#     "baseline_annual_pop_growth_rate": 0.001,
#     "baseline_annual_gdp_growth_rate": 0.012,
#     "baseline_annual_productivity_growth_rate": 0.01,
#     "poverty_floor": 0.03
# }


# --- Dicionário de Formatação ---
# This will be updated later based on UBISimulatorRealisticV3's output columns
# formatters_dict_old = {
#     "Population (Millions)": "{:.1f}",
#     "UBI Cost (Billion EUR)": "{:.1f}",
#     "Labor Participation Rate": "{:.1%}",
#     "Avg Hours Worked": "{:.0f}",
#     "Unemployment Rate": "{:.1%}",
#     "Labor Force (Millions)": "{:.1f}",
#     "GDP Real (Trillion EUR)": "{:.2f}",
#     "GDP Growth Rate": "{:.2%}",
#     "Productivity Growth Rate": "{:.2%}",
#     "Productivity Level (Index)": "{:.1f}",
#     "Consumption (Trillion EUR)": "{:.2f}",
#     "Investment (Trillion EUR)": "{:.2f}",
#     "Govt Debt/GDP Ratio": "{:.1%}",
#     "Additional Tax Revenue (Billion EUR)": "{:.1f}",
#     "Capital Flight Index": "{:.1f}",
#     "Gini Coefficient": "{:.3f}",
#     "Poverty Rate": "{:.1%}",
#     "Wellbeing Satisfaction (0-10)": "{:.1f}",
#     "Mental Health Prevalence": "{:.2%}",
#     "Crime Rate Index": "{:.1f}",
#     "Unpaid Work Index": "{:.1f}",
#     "Net Migration (Thousands)": "{:.0f}",
#     "Millionaires (Thousands)": "{:.0f}",
#     "Inflation Rate": "{:.2%}"
# }

# Placeholder for new formatters based on UBISimulatorRealisticV3 output
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
    # Cohort specific columns will be numerous, so not listing them here for individual formatting
    # but they typically end with '(M)' for population or are rates.
}


# --- Classe UBISimulator (sem alterações na lógica interna) ---
# class UBISimulator:
#     # ... (código da classe exatamente como na resposta anterior) ...
#     # ... (_initialize_scenario, run_simulation, get_results, get_summary_dataframe) ...
#     """
#     Simulador de Impactos de Longo Prazo da Renda Básica Universal (RBU).
#     Modela cenários 'Com RBU' vs. 'Sem RBU' (Baseline) ao longo do tempo.
#     """
#     def __init__(self, params=None):
#         if params is None:
#              params = DEFAULT_PARAMS.copy()
#         # Garante tipos numéricos corretos ao receber params
#         self.params = {}
#         for k, v in params.items():
#              # Heurística simples para tipos
#              if isinstance(v, bool): # Preserva booleanos se houver
#                  self.params[k] = v
#              elif k in ['start_year', 'end_year', 'ubi_start_year']:
#                  try: self.params[k] = int(v)
#                  except (ValueError, TypeError): self.params[k] = int(DEFAULT_PARAMS[k]) # Fallback
#              elif k == 'financing_model':
#                  self.params[k] = str(v)
#              else: # Tenta converter outros para float
#                  try: self.params[k] = float(v)
#                  except (ValueError, TypeError): self.params[k] = v # Mantem original se falhar

#         # Garante que end_year seja pelo menos start_year (como int)
#         self.params["end_year"] = max(int(self.params["start_year"]), int(self.params["end_year"]))
#         self.years = np.arange(int(self.params["start_year"]), int(self.params["end_year"]) + 1)
#         self.results = {}

#     def _initialize_scenario(self):
#         """Cria um DataFrame para armazenar os resultados anuais de um cenário."""
#         indicators = list(formatters_dict.keys()) # Usa as chaves do formatador como lista de indicadores
#         index_years = self.years if len(self.years) > 0 else [int(self.params["start_year"])]
#         df = pd.DataFrame(index=index_years, columns=indicators, dtype=float)

#         # Preenche valores iniciais para o ano base (start_year - 1) para cálculos
#         base_year = int(self.params["start_year"]) - 1
#         # Garante que o índice base existe antes de tentar preencher
#         if base_year not in df.index:
#              # Adiciona linha NaN se não existir, garantindo que seja float
#              df.loc[base_year] = np.full(len(indicators), np.nan, dtype=float)


#         df.loc[base_year, "Population (Millions)"] = float(self.params["initial_population_millions"])
#         df.loc[base_year, "GDP Real (Trillion EUR)"] = float(self.params["initial_gdp_trillion_eur"])
#         df.loc[base_year, "Labor Participation Rate"] = float(self.params["initial_participation_rate"])
#         df.loc[base_year, "Avg Hours Worked"] = float(self.params["initial_avg_hours_worked_per_worker"])
#         df.loc[base_year, "Unemployment Rate"] = float(self.params["initial_unemployment_rate"])
#         df.loc[base_year, "Productivity Level (Index)"] = 100.0 # Base 100
#         df.loc[base_year, "Gini Coefficient"] = float(self.params["initial_gini_coefficient"])
#         df.loc[base_year, "Poverty Rate"] = float(self.params["initial_poverty_rate"])
#         df.loc[base_year, "Wellbeing Satisfaction (0-10)"] = float(self.params["initial_wellbeing_satisfaction"])
#         df.loc[base_year, "Mental Health Prevalence"] = float(self.params["initial_mental_health_prevalence"])
#         df.loc[base_year, "Crime Rate Index"] = float(self.params["initial_crime_rate_index"])
#         df.loc[base_year, "Net Migration (Thousands)"] = float(self.params["initial_net_migration_thousands"])
#         df.loc[base_year, "Millionaires (Thousands)"] = float(self.params["initial_millionaires_thousands"])
#         df.loc[base_year, "Govt Debt/GDP Ratio"] = float(self.params["initial_govt_debt_gdp_ratio"])
#         df.loc[base_year, "Capital Flight Index"] = 100.0
#         df.loc[base_year, "Unpaid Work Index"] = 100.0
#         df.loc[base_year, "Inflation Rate"] = float(self.params["baseline_annual_inflation_rate"])

#         # Valores derivados iniciais
#         pop_base = df.loc[base_year, "Population (Millions)"]
#         part_rate_base = df.loc[base_year, "Labor Participation Rate"]
#         unemp_rate_base = df.loc[base_year, "Unemployment Rate"]
#         if pd.notna(pop_base) and pop_base > 0 and pd.notna(part_rate_base) and pd.notna(unemp_rate_base):
#             initial_labor_force = pop_base * part_rate_base
#             df.loc[base_year, "Labor Force (Millions)"] = initial_labor_force * (1.0 - unemp_rate_base) # Empregados
#         else:
#             df.loc[base_year, "Labor Force (Millions)"] = 0.0

#         gdp_base = df.loc[base_year, "GDP Real (Trillion EUR)"]
#         if pd.notna(gdp_base) and gdp_base > 0:
#             df.loc[base_year, "Consumption (Trillion EUR)"] = gdp_base * 0.6
#             df.loc[base_year, "Investment (Trillion EUR)"] = gdp_base * 0.2
#         else:
#             df.loc[base_year, "Consumption (Trillion EUR)"] = 0.0
#             df.loc[base_year, "Investment (Trillion EUR)"] = 0.0

#         # Inicializa outras colunas que podem não ter valor base explícito
#         df.loc[base_year, "UBI Cost (Billion EUR)"] = 0.0
#         df.loc[base_year, "GDP Growth Rate"] = 0.0
#         # Garante que Productivity Growth Rate tenha valor inicial
#         if pd.isna(df.loc[base_year, "Productivity Growth Rate"]):
#              df.loc[base_year, "Productivity Growth Rate"] = float(self.params["baseline_annual_productivity_growth_rate"])
#         df.loc[base_year, "Additional Tax Revenue (Billion EUR)"] = 0.0

#         return df

#     def run_simulation(self):
#         """Executa a simulação para os cenários Baseline e Com RBU."""
#         start_year_int = int(self.params["start_year"])
#         base_year = start_year_int - 1

#         if len(self.years) == 0:
#              # Handle case where start_year == end_year
#              initialized_df = self._initialize_scenario()
#              # Check if base_year exists, otherwise use start_year_int
#              year_to_use = base_year if base_year in initialized_df.index else start_year_int
#              if year_to_use not in initialized_df.index: # Failsafe if neither exists
#                   st.error(f"Cannot initialize simulation for year {start_year_int}")
#                   return

#              single_year_data = initialized_df.loc[[year_to_use]].rename(index={year_to_use: start_year_int})
#              baseline_df = single_year_data.fillna(0.0)
#              ubi_df = single_year_data.fillna(0.0) # UBI won't be active anyway
#              self.results["Baseline (Sem RBU)"] = baseline_df
#              self.results["Com RBU"] = ubi_df
#              return


#         # --- Cenário Baseline (Sem RBU) ---
#         baseline_df = self._initialize_scenario()
#         for year in self.years:
#             prev_year = year - 1
#             if prev_year not in baseline_df.index: continue # Skip if previous year data is missing

#             # Copia valores do ano anterior como base para muitos indicadores baseline
#             baseline_df.loc[year] = baseline_df.loc[prev_year].copy()

#             # --- Cálculos do Ano Atual (Baseline) ---
#             prev_pop = baseline_df.loc[prev_year, "Population (Millions)"]
#             prev_mig = baseline_df.loc[prev_year, "Net Migration (Thousands)"]
#             baseline_df.loc[year, "Population (Millions)"] = prev_pop * (1.0 + self.params["baseline_annual_pop_growth_rate"]) + (prev_mig / 1000.0)

#             prev_prod_level = baseline_df.loc[prev_year, "Productivity Level (Index)"]
#             baseline_df.loc[year, "Productivity Growth Rate"] = self.params["baseline_annual_productivity_growth_rate"]
#             baseline_df.loc[year, "Productivity Level (Index)"] = prev_prod_level * (1.0 + baseline_df.loc[year, "Productivity Growth Rate"])

#             current_pop = baseline_df.loc[year, "Population (Millions)"]
#             current_part_rate = baseline_df.loc[year, "Labor Participation Rate"] # Mantido do ano anterior
#             current_unemp_rate = baseline_df.loc[year, "Unemployment Rate"] # Mantido do ano anterior
#             if pd.notna(current_pop) and pd.notna(current_part_rate) and pd.notna(current_unemp_rate) and current_pop > 0:
#                 potential_labor_force = current_pop * current_part_rate
#                 employed_labor_force = potential_labor_force * (1.0 - current_unemp_rate)
#                 baseline_df.loc[year, "Labor Force (Millions)"] = employed_labor_force
#             else:
#                  baseline_df.loc[year, "Labor Force (Millions)"] = 0.0

#             gdp_prev = baseline_df.loc[prev_year, "GDP Real (Trillion EUR)"]
#             prev_labor = baseline_df.loc[prev_year, "Labor Force (Millions)"]
#             current_labor = baseline_df.loc[year, "Labor Force (Millions)"]
#             prev_hours_b = baseline_df.loc[prev_year, "Avg Hours Worked"]
#             current_hours_b = baseline_df.loc[year, "Avg Hours Worked"] # Mantido do ano anterior
#             current_prod_growth_b = baseline_df.loc[year, "Productivity Growth Rate"]

#             # Calcula GDP baseado em trabalho e produtividade
#             gdp_current = 0.0
#             if pd.notna(gdp_prev) and gdp_prev > 0:
#                  labor_input_prev = prev_labor * prev_hours_b
#                  labor_input_curr = current_labor * current_hours_b
#                  if pd.notna(labor_input_prev) and labor_input_prev > 0 and pd.notna(labor_input_curr) and pd.notna(current_prod_growth_b):
#                      labor_input_growth = (labor_input_curr / labor_input_prev) - 1.0
#                      gdp_current = gdp_prev * (1.0 + labor_input_growth + current_prod_growth_b)
#                  else: # Fallback se dados de trabalho ausentes
#                      gdp_current = gdp_prev * (1.0 + self.params["baseline_annual_gdp_growth_rate"])

#             baseline_df.loc[year, "GDP Real (Trillion EUR)"] = gdp_current if gdp_current > 0 else 0.0
#             baseline_df.loc[year, "GDP Growth Rate"] = (gdp_current / gdp_prev - 1.0) if pd.notna(gdp_prev) and gdp_prev > 0 else 0.0

#             # --- Baseline Social/Outros ---
#             baseline_df.loc[year, "Millionaires (Thousands)"] *= 1.02
#             if pd.notna(gdp_current) and gdp_current > 0 and pd.notna(gdp_prev) and gdp_prev > 0:
#                  baseline_df.loc[year, "Govt Debt/GDP Ratio"] = baseline_df.loc[prev_year, "Govt Debt/GDP Ratio"] * (gdp_prev / gdp_current)
#             # else: Mantem a dívida do ano anterior se GDP for zero ou NaN

#             baseline_df.loc[year, "Capital Flight Index"] = 100.0
#             baseline_df.loc[year, "UBI Cost (Billion EUR)"] = 0.0
#             baseline_df.loc[year, "Additional Tax Revenue (Billion EUR)"] = 0.0

#             # --- Baseline Consumption/Investment ---
#             prev_cons_b = baseline_df.loc[prev_year, "Consumption (Trillion EUR)"]
#             prev_inv_b = baseline_df.loc[prev_year, "Investment (Trillion EUR)"]
#             gdp_growth_b = baseline_df.loc[year, "GDP Growth Rate"]
#             if pd.notna(gdp_current) and gdp_current > 0 and pd.notna(prev_cons_b) and pd.notna(prev_inv_b):
#                 baseline_df.loc[year, "Consumption (Trillion EUR)"] = prev_cons_b * (1.0 + gdp_growth_b)
#                 baseline_df.loc[year, "Investment (Trillion EUR)"] = prev_inv_b * (1.0 + gdp_growth_b)
#             else:
#                 baseline_df.loc[year, "Consumption (Trillion EUR)"] = 0.0
#                 baseline_df.loc[year, "Investment (Trillion EUR)"] = 0.0


#             # --- Baseline Inflation ---
#             prod_growth_b = baseline_df.loc[year, "Productivity Growth Rate"]
#             inflation_reduction_from_prod_b = prod_growth_b * self.params["inflation_productivity_dampening_factor"]
#             baseline_inflation = self.params["baseline_annual_inflation_rate"] - inflation_reduction_from_prod_b
#             baseline_df.loc[year, "Inflation Rate"] = max(0.0, baseline_inflation)

#         # Remove base_year e preenche NaNs remanescentes
#         self.results["Baseline (Sem RBU)"] = baseline_df.drop(index=base_year, errors='ignore').fillna(0.0)


#         # --- Cenário Com RBU ---
#         ubi_df = self._initialize_scenario()
#         for year in self.years:
#             prev_year = year - 1
#             if prev_year not in ubi_df.index: continue # Skip if previous year data is missing

#             ubi_active = year >= int(self.params["ubi_start_year"])
#             # Copia valores do ano anterior como base
#             ubi_df.loc[year] = ubi_df.loc[prev_year].copy()

#             # --- Cálculos do Ano Atual (Com RBU) ---
#             migration_factor = self.params["migration_net_increase_factor_restricted"] if ubi_active else 1.0
#             net_migration_ubi = self.params["initial_net_migration_thousands"] * migration_factor
#             ubi_df.loc[year, "Net Migration (Thousands)"] = net_migration_ubi

#             prev_pop_ubi = ubi_df.loc[prev_year, "Population (Millions)"]
#             fertility_boost = self.params['fertility_rate_ubi_boost_pp'] if ubi_active else 0.0
#             ubi_df.loc[year, "Population (Millions)"] = prev_pop_ubi * (1.0 + self.params["baseline_annual_pop_growth_rate"] + fertility_boost) + (net_migration_ubi / 1000.0)

#             ubi_cost = 0.0
#             current_pop_ubi = ubi_df.loc[year, "Population (Millions)"]
#             if ubi_active and pd.notna(current_pop_ubi) and current_pop_ubi > 0 and self.params["ubi_annual_amount_eur"] > 0:
#                 eligible_pop = current_pop_ubi * self.params["ubi_eligible_population_share"]
#                 ubi_cost = (eligible_pop * 1_000_000.0 * self.params["ubi_annual_amount_eur"]) / 1_000_000_000.0
#             ubi_df.loc[year, "UBI Cost (Billion EUR)"] = ubi_cost

#             # --- Mercado de Trabalho (Com RBU) ---
#             participation_rate_base_ubi = ubi_df.loc[prev_year, "Labor Participation Rate"]
#             avg_hours_base_ubi = ubi_df.loc[prev_year, "Avg Hours Worked"]
#             ubi_df.loc[year, "Labor Participation Rate"] = participation_rate_base_ubi * (1.0 - self.params["labor_participation_reduction_factor"]) if ubi_active else participation_rate_base_ubi
#             ubi_df.loc[year, "Avg Hours Worked"] = avg_hours_base_ubi * (1.0 - self.params["labor_hours_reduction_factor"]) if ubi_active else avg_hours_base_ubi

#             # --- Produtividade (Com RBU) ---
#             productivity_boost = self.params["productivity_growth_boost_pp"] if ubi_active else 0.0
#             ubi_df.loc[year, "Productivity Growth Rate"] = self.params["baseline_annual_productivity_growth_rate"] + productivity_boost
#             ubi_df.loc[year, "Productivity Level (Index)"] = ubi_df.loc[prev_year, "Productivity Level (Index)"] * (1.0 + ubi_df.loc[year, "Productivity Growth Rate"])

#             # --- Força de Trabalho (Com RBU) ---
#             current_part_rate_ubi = ubi_df.loc[year, "Labor Participation Rate"]
#             current_unemp_rate_ubi = ubi_df.loc[year, "Unemployment Rate"] # Mantido baseline
#             if pd.notna(current_pop_ubi) and pd.notna(current_part_rate_ubi) and pd.notna(current_unemp_rate_ubi) and current_pop_ubi > 0:
#                  potential_labor_force_ubi = current_pop_ubi * current_part_rate_ubi
#                  employed_labor_force_ubi = potential_labor_force_ubi * (1.0 - current_unemp_rate_ubi)
#                  ubi_df.loc[year, "Labor Force (Millions)"] = employed_labor_force_ubi
#             else:
#                  ubi_df.loc[year, "Labor Force (Millions)"] = 0.0

#             # --- GDP (Com RBU) ---
#             gdp_prev_ubi = ubi_df.loc[prev_year, "GDP Real (Trillion EUR)"]
#             prev_labor_ubi = ubi_df.loc[prev_year, "Labor Force (Millions)"]
#             prev_hours_ubi = ubi_df.loc[prev_year, "Avg Hours Worked"]
#             current_prod_growth_ubi = ubi_df.loc[year, "Productivity Growth Rate"]
#             current_labor_ubi = ubi_df.loc[year, "Labor Force (Millions)"]
#             current_hours_ubi = ubi_df.loc[year, "Avg Hours Worked"]

#             # --> Variável unificada é gdp_current_ubi <--
#             gdp_current_ubi = 0.0 # Inicializa
#             if pd.notna(gdp_prev_ubi) and gdp_prev_ubi > 0:
#                  labor_input_prev_ubi = prev_labor_ubi * prev_hours_ubi
#                  labor_input_curr_ubi = current_labor_ubi * current_hours_ubi
#                  if pd.notna(labor_input_prev_ubi) and labor_input_prev_ubi > 0 and pd.notna(labor_input_curr_ubi) and pd.notna(current_prod_growth_ubi):
#                      labor_input_growth_ubi = (labor_input_curr_ubi / labor_input_prev_ubi) - 1.0
#                      gdp_current_ubi = gdp_prev_ubi * (1.0 + labor_input_growth_ubi + current_prod_growth_ubi)
#                  else: # Fallback
#                      gdp_current_ubi = gdp_prev_ubi * (1.0 + self.params["baseline_annual_gdp_growth_rate"])

#             ubi_df.loc[year, "GDP Real (Trillion EUR)"] = gdp_current_ubi if gdp_current_ubi > 0 else 0.0
#             ubi_df.loc[year, "GDP Growth Rate"] = (gdp_current_ubi / gdp_prev_ubi - 1.0) if pd.notna(gdp_prev_ubi) and gdp_prev_ubi > 0 else 0.0

#             # --- Financiamento e Dívida (Com RBU) ---
#             tax_revenue = 0.0
#             debt_increase_ratio = 0.0
#             # --> Usa gdp_current_ubi consistentemente <--
#             if ubi_active and pd.notna(gdp_current_ubi) and gdp_current_ubi > 0 and ubi_cost > 0:
#                 if self.params["financing_model"] in ["progressive_tax", "wealth_tax", "mixed_taxes_debt"]:
#                     tax_revenue = gdp_current_ubi * 1000.0 * self.params["additional_tax_rate_for_ubi"]
#                 ubi_df.loc[year, "Additional Tax Revenue (Billion EUR)"] = tax_revenue
#                 unfunded_cost = ubi_cost - tax_revenue
#                 # Permite dívida diminuir se houver superávit (unfunded_cost < 0)
#                 if self.params["financing_model"] in ["debt", "mixed_taxes_debt"]:
#                     debt_increase_ratio = (unfunded_cost / 1000.0) / gdp_current_ubi
#             else:
#                  ubi_df.loc[year, "Additional Tax Revenue (Billion EUR)"] = 0.0

#             prev_debt_ratio_ubi = ubi_df.loc[prev_year, "Govt Debt/GDP Ratio"]
#             current_debt_ratio = prev_debt_ratio_ubi # Default value
#             if pd.notna(gdp_current_ubi) and gdp_current_ubi > 0:
#                  if pd.notna(gdp_prev_ubi) and gdp_prev_ubi > 0:
#                      current_debt_ratio = (prev_debt_ratio_ubi * gdp_prev_ubi / gdp_current_ubi) + debt_increase_ratio
#                  else: # Caso do primeiro ano ou se GDP anterior for zero
#                      current_debt_ratio = prev_debt_ratio_ubi + debt_increase_ratio
#             # Se gdp_current_ubi for 0 ou NaN, mantém a razão anterior (evita divisão por zero)

#             ubi_df.loc[year, "Govt Debt/GDP Ratio"] = max(0.0, current_debt_ratio)

#             # --- Consumo e Investimento (Com RBU) ---
#             prev_consumption_ubi = ubi_df.loc[prev_year, "Consumption (Trillion EUR)"]
#             prev_investment_ubi = ubi_df.loc[prev_year, "Investment (Trillion EUR)"]
#             gdp_growth_ubi = ubi_df.loc[year, "GDP Growth Rate"]
#             consumption_ubi = 0.0
#             investment_ubi = 0.0
#             if pd.notna(prev_consumption_ubi) and pd.notna(prev_investment_ubi):
#                  consumption_ubi = prev_consumption_ubi * (1.0 + gdp_growth_ubi)
#                  investment_ubi = prev_investment_ubi * (1.0 + gdp_growth_ubi)

#             net_consumption_boost_trillions = 0.0
#             if ubi_active and ubi_cost > 0:
#                 consumption_boost_from_ubi = ubi_cost * self.params["consumption_propensity_ubi_recipients"]
#                 consumption_reduction_from_taxes = (tax_revenue * self.params["consumption_reduction_high_income_financing"]) if self.params["financing_model"] != "debt" else 0.0
#                 net_consumption_boost_billions = consumption_boost_from_ubi - consumption_reduction_from_taxes
#                 net_consumption_boost_trillions = net_consumption_boost_billions / 1000.0
#                 consumption_ubi += net_consumption_boost_trillions
#                 if self.params["financing_model"] != "debt":
#                      investment_ubi *= (1.0 - self.params["additional_tax_rate_for_ubi"] * 0.1)

#             ubi_df.loc[year, "Consumption (Trillion EUR)"] = max(0.0, consumption_ubi)
#             ubi_df.loc[year, "Investment (Trillion EUR)"] = max(0.0, investment_ubi)

#             # --- Indicadores Sociais (Com RBU) ---
#             gini_base_ubi = ubi_df.loc[prev_year, "Gini Coefficient"]
#             poverty_base_ubi = ubi_df.loc[prev_year, "Poverty Rate"]
#             wellbeing_base_ubi = ubi_df.loc[prev_year, "Wellbeing Satisfaction (0-10)"]
#             mental_health_base_ubi = ubi_df.loc[prev_year, "Mental Health Prevalence"]
#             crime_base_ubi = ubi_df.loc[prev_year, "Crime Rate Index"]
#             unpaid_work_base_ubi = ubi_df.loc[prev_year, "Unpaid Work Index"]

#             if ubi_active:
#                 ubi_amount_effect = self.params["ubi_annual_amount_eur"] / 14400.0 if self.params["ubi_annual_amount_eur"] > 0 else 0.0
#                 gini_reduction_max = 0.04
#                 gini_reduction = gini_reduction_max * ubi_amount_effect
#                 gini_ubi = max(0.0, gini_base_ubi - gini_reduction)
#                 ubi_df.loc[year, "Gini Coefficient"] = gini_ubi

#                 poverty_reduction_factor = 0.9
#                 poverty_ubi = poverty_base_ubi * (1.0 - poverty_reduction_factor * ubi_amount_effect)
#                 ubi_df.loc[year, "Poverty Rate"] = max(self.params.get("poverty_floor", 0.03), poverty_ubi)

#                 ubi_df.loc[year, "Wellbeing Satisfaction (0-10)"] = min(10.0, wellbeing_base_ubi + self.params["wellbeing_satisfaction_boost"] * ubi_amount_effect)

#                 mental_health_reduction = self.params["mental_health_reduction_pct"] / 100.0
#                 prevalence_factor = (1.0 - mental_health_reduction)
#                 effective_factor = 1.0 - (1.0 - prevalence_factor) * ubi_amount_effect
#                 ubi_df.loc[year, "Mental Health Prevalence"] = mental_health_base_ubi * effective_factor

#                 gini_change = max(0.0, gini_base_ubi - gini_ubi) # Garante não negativo
#                 crime_reduction_factor = (gini_change / 0.01) * self.params["crime_reduction_per_gini_point"] if gini_change > 0 else 0.0
#                 ubi_df.loc[year, "Crime Rate Index"] = max(0.0, crime_base_ubi * (1.0 - crime_reduction_factor))

#                 ubi_df.loc[year, "Unpaid Work Index"] = unpaid_work_base_ubi * self.params["unpaid_work_increase_factor"]

#             # --- Riqueza e Fuga de Capital (Com RBU) ---
#             millionaires_base_ubi = ubi_df.loc[prev_year, "Millionaires (Thousands)"]
#             capital_flight_base_ubi = ubi_df.loc[prev_year, "Capital Flight Index"]
#             millionaires_ubi = millionaires_base_ubi * 1.02 # Crescimento base
#             capital_flight_ubi = capital_flight_base_ubi # Mantem base se não houver fuga

#             if ubi_active and self.params["financing_model"] != "debt" and self.params["capital_flight_factor_high_tax"] > 0:
#                 flight_increase_factor = 1.0 + self.params["capital_flight_factor_high_tax"]
#                 capital_flight_ubi = capital_flight_base_ubi * flight_increase_factor
#                 millionaires_ubi *= (1.0 - self.params["capital_flight_factor_high_tax"])

#             ubi_df.loc[year, "Millionaires (Thousands)"] = max(0.0, millionaires_ubi)
#             ubi_df.loc[year, "Capital Flight Index"] = capital_flight_ubi if capital_flight_ubi > 0 else 0.0

#             # --- Inflação (Com RBU) ---
#             current_inflation = self.params["baseline_annual_inflation_rate"]
#             current_prod_growth_ubi = ubi_df.loc[year, "Productivity Growth Rate"]
#             if pd.notna(current_prod_growth_ubi):
#                  inflation_reduction_from_prod = current_prod_growth_ubi * self.params["inflation_productivity_dampening_factor"]
#                  current_inflation -= inflation_reduction_from_prod

#             # --> Usa gdp_current_ubi consistentemente <--
#             if ubi_active and pd.notna(gdp_current_ubi) and gdp_current_ubi > 0 and net_consumption_boost_trillions > 0:
#                 ubi_consumption_boost_ratio = net_consumption_boost_trillions / gdp_current_ubi
#                 inflation_boost_from_demand = ubi_consumption_boost_ratio * self.params["inflation_ubi_demand_sensitivity"]
#                 current_inflation += inflation_boost_from_demand

#             ubi_df.loc[year, "Inflation Rate"] = max(0.0, current_inflation)

#         # Remove base_year e preenche NaNs remanescentes
#         self.results["Com RBU"] = ubi_df.drop(index=base_year, errors='ignore').fillna(0.0)


    def get_results(self):
        """Retorna os DataFrames de resultados para ambos os cenários."""
        return self.results

    def get_summary_dataframe(self, year):
        """Retorna um DataFrame NÃO FORMATADO para a tabela comparativa de um ano específico."""
        if not self.results or "Baseline (Sem RBU)" not in self.results or "Com RBU" not in self.results :
             return pd.DataFrame(columns=["Cenário Sem RBU", "Cenário Com RBU"], index=pd.Index([], name="Indicador"))

        # Garante que year é int
        target_year = int(max(int(self.params["start_year"]), min(int(year), int(self.params["end_year"]))))

        if target_year not in self.results["Baseline (Sem RBU)"].index or target_year not in self.results["Com RBU"].index:
             # st.error(f"Ano {target_year} fora do intervalo da simulação ({self.params['start_year']}-{self.params['end_year']}) ou dados ausentes.")
             # Retorna DataFrame vazio com índice para evitar erros posteriores
             return pd.DataFrame(columns=["Cenário Sem RBU", "Cenário Com RBU"], index=pd.Index([], name="Indicador"))

        baseline_data = self.results["Baseline (Sem RBU)"].loc[target_year]
        ubi_data = self.results["Com RBU"].loc[target_year]

        # Garante que os índices sejam os mesmos para evitar desalinhamento
        common_index = baseline_data.index.intersection(ubi_data.index)
        summary = pd.DataFrame({
            "Indicador": common_index, # Usa apenas indicadores comuns
            "Cenário Sem RBU": baseline_data.loc[common_index].values,
            "Cenário Com RBU": ubi_data.loc[common_index].values
        })

        return summary.set_index("Indicador") # Retorna DF com dados numéricos



# --- Interface Streamlit ---

st.set_page_config(layout="wide")
st.title("Simulador Interativo de Impactos da Renda Básica Universal (RBU)")
st.markdown("""
Use a barra lateral para ajustar **todas** as premissas da simulação, agrupadas por categoria.
Analise os impactos econômicos, sociais e fiscais de longo prazo comparando o cenário 'Com RBU' vs. 'Sem RBU' (Baseline).
**Nota:** Este é um modelo simplificado para fins ilustrativos. Os resultados dependem fortemente das premissas escolhidas.
""")

# --- Barra Lateral para Controles ---
st.sidebar.header("⚙️ Parâmetros da Simulação")

# Use a deep copy of DEFAULT_SCENARIO_PARAMS to avoid modifying the original
# For UI state management, it's often better to initialize directly from UI elements
# or have a separate state management for UI if it becomes complex.
# For now, we'll initialize a dictionary and populate it.
ui_params = {} 

# --- Grupo: Configuração da Simulação (AGORA COM INPUTS) ---
with st.sidebar.expander("⏱️ Configuração da Simulação", expanded=True):
    
    # Country Selection
    country_selected = st.selectbox(
        "País:", 
        ["germany", "france"], 
        index=["germany", "france"].index(DEFAULT_SCENARIO_PARAMS["country"]), # Default from imported params
        key="country_select"
    )
    ui_params["country"] = country_selected

    # Simulation years, default from DEFAULT_SCENARIO_PARAMS for the selected/default country
    default_sim_years = DEFAULT_SCENARIO_PARAMS.get("simulation_years", {"start": datetime.date.today().year, "end": datetime.date.today().year + 50})
    default_start = int(default_sim_years["start"])
    default_end = int(default_sim_years["end"])

    sim_start_year = st.number_input(
        "Ano Inicial da Simulação",
        min_value=2020,
        max_value=default_end - 1, 
        value=default_start,
        step=1,
        key=f"sim_start_year_input_{country_selected}",
        help="O primeiro ano a ser simulado."
    )
    sim_end_year = st.number_input(
        "Ano Final da Simulação",
        min_value=sim_start_year, 
        max_value=2150,
        value=default_end,
        step=1,
        key=f"sim_end_year_input_{country_selected}",
        help="O último ano a ser simulado."
    )

    if sim_end_year < sim_start_year:
        st.sidebar.error("O ano final deve ser maior ou igual ao ano inicial.")
        sim_start_year = default_start
        sim_end_year = default_end
    
    ui_params["simulation_years"] = {"start": int(sim_start_year), "end": int(sim_end_year)}

# --- Grupo: Configuração da RBU ---
with st.sidebar.expander("🔵 Configuração da RBU", expanded=True):
    default_ubi_params = DEFAULT_SCENARIO_PARAMS.get("ubi", {})
    ui_params["ubi"] = {} # Initialize sub-dictionary
    
    ui_params["ubi"]["annual_amount_eur_real_start"] = st.slider(
        "Valor Anual REAL Inicial RBU (€)", 0, 25000, 
        int(default_ubi_params.get("annual_amount_eur_real_start", 14400)), 
        500, format="€%.0f", key=f"ubi_amount_{country_selected}"
    )
    
    _sim_start = ui_params["simulation_years"]["start"]
    _sim_end = ui_params["simulation_years"]["end"]
    _default_ubi_start = int(default_ubi_params.get("start_year", _sim_start + 3))
    ubi_start_value = max(_sim_start, min(_sim_end, _default_ubi_start))

    ui_params["ubi"]["start_year"] = st.slider(
        "Ano de Início da RBU",
        min_value=_sim_start,
        max_value=_sim_end,
        value=ubi_start_value,
        step=1,
        key=f"ubi_start_{country_selected}"
    )
    # Add other UBI params if they are to be configurable, e.g.:
    # ui_params["ubi"]["min_eligible_age"] = st.slider("Idade Mínima Elegibilidade RBU", 0, 25, int(default_ubi_params.get("min_eligible_age", 18)), 1, key=f"ubi_min_age_{country_selected}")
    # ui_params["ubi"]["indexation_lag"] = st.slider("Lag Indexação RBU (anos)", 0, 3, int(default_ubi_params.get("indexation_lag", 1)), 1, key=f"ubi_index_lag_{country_selected}")
    ui_params["ubi"]["fertility_rate_boost_factor"] = st.slider(
        "Aumento Taxa Fertilidade com RBU (fator)", 0.0, 0.1, 
        float(default_ubi_params.get("fertility_rate_boost_factor", 0.02)), 
        0.005, format="%.3f", key=f"ubi_fert_boost_{country_selected}"
    )


# --- Grupo: Mercado de Trabalho ---
with st.sidebar.expander("💼 Mercado de Trabalho"):
    default_labor_params = DEFAULT_SCENARIO_PARAMS.get("labor_market", {})
    ui_params["labor_market"] = {} # Initialize sub-dictionary
    
    # Nested UBI participation reduction factor
    default_part_reduc = default_labor_params.get("ubi_participation_reduction_factor", {"working_age": 0.05, "old_age": 0.10})
    ui_params["labor_market"]["ubi_participation_reduction_factor"] = default_part_reduc.copy() # Start with defaults

    ui_params["labor_market"]["ubi_participation_reduction_factor"]["working_age"] = st.slider(
        "Redução Participação (Idade Ativa, %)", 0.0, 15.0, 
        float(default_part_reduc.get("working_age", 0.05)) * 100, 
        0.5, format="%.1f%%", key=f"labor_reduc_active_{country_selected}"
    ) / 100.0
    
    ui_params["labor_market"]["ubi_hours_reduction_factor"] = st.slider(
        "Redução Horas Trabalhadas (Idade Ativa, %)", 0.0, 10.0, 
        float(default_labor_params.get("ubi_hours_reduction_factor", 0.01)) * 100, 
        0.1, format="%.1f%%", key=f"hours_reduc_{country_selected}"
    ) / 100.0
    # NAIRU is loaded from data, placeholder_avg_hours could be an advanced setting if needed

# --- Grupo: Economia (Produtividade, Inflação) ---
with st.sidebar.expander("📈 Economia"):
    default_econ_params = DEFAULT_SCENARIO_PARAMS.get("economy", {})
    ui_params["economy"] = {}

    ui_params["economy"]["ubi_tfp_growth_boost_pp"] = st.slider(
        "Boost TFP Anual com RBU (p.p.)", 0.0, 0.5, 
        float(default_econ_params.get("ubi_tfp_growth_boost_pp", 0.001)) * 100, 
        0.05, format="%.2f p.p.", key=f"tfp_boost_{country_selected}"
    ) / 100.0
    ui_params["economy"]["inflation_ubi_demand_sensitivity"] = st.slider(
        "Sensibilidade Inflação -> Demanda RBU", 0.0, 1.0, 
        float(default_econ_params.get("inflation_ubi_demand_sensitivity", 0.3)), 
        0.05, key=f"inf_sens_demand_{country_selected}"
    )
    ui_params["economy"]["inflation_output_gap_sensitivity"] = st.slider(
        "Sensibilidade Inflação -> Hiato do Produto", 0.0, 1.0,
        float(default_econ_params.get("inflation_output_gap_sensitivity", 0.5)),
        0.05, key=f"inf_sens_gap_{country_selected}"
    )
    # Other economy params like baseline_tfp_growth_rate, capital_depreciation_rate, etc.,
    # are kept as defaults from DEFAULT_SCENARIO_PARAMS unless added to UI.

# --- Grupo: Financiamento RBU ---
with st.sidebar.expander("💰 Financiamento RBU"):
    default_finance_params = DEFAULT_SCENARIO_PARAMS.get("government_finance", {})
    ui_params["government_finance"] = {}

    ui_params["government_finance"]["ubi_financing_tax_share"] = st.slider(
        "Parcela da RBU Financiada por Impostos Adicionais (%)", 0.0, 100.0,
        float(default_finance_params.get("ubi_financing_tax_share", 0.7)) * 100,
        5.0, format="%.0f%%", key=f"ubi_tax_share_{country_selected}"
    ) / 100.0
    ui_params["government_finance"]["consumption_propensity_ubi_recipients"] = st.slider(
        "Propensão a Consumir (Beneficiários RBU, %)", 0.0, 100.0,
        float(default_finance_params.get("consumption_propensity_ubi_recipients", 0.8)) * 100,
        5.0, format="%.0f%%", key=f"consump_prop_ubi_{country_selected}"
    ) / 100.0
    ui_params["government_finance"]["consumption_reduction_high_income_financing"] = st.slider(
        "Redução Consumo (Financiadores-Renda Alta, % do imposto pago)", 0.0, 50.0,
        float(default_finance_params.get("consumption_reduction_high_income_financing", 0.2)) * 100,
        5.0, format="%.0f%%", key=f"consump_reduc_fin_{country_selected}"
    ) / 100.0

# --- Construct final_params by merging UI inputs with DEFAULT_SCENARIO_PARAMS ---
# This ensures all necessary parameters are present.
final_params = DEFAULT_SCENARIO_PARAMS.copy() # Start with a full copy of defaults
# Deep copy for nested dictionaries
for key, value in DEFAULT_SCENARIO_PARAMS.items():
    if isinstance(value, dict):
        final_params[key] = value.copy()
        for sub_key, sub_value in value.items():
            if isinstance(sub_value, dict): # Handle two levels of nesting
                 final_params[key][sub_key] = sub_value.copy()


# Update with UI-set parameters
final_params["country"] = ui_params["country"]
final_params["simulation_years"].update(ui_params.get("simulation_years", {}))
final_params["ubi"].update(ui_params.get("ubi", {}))
final_params["labor_market"].update(ui_params.get("labor_market", {}))
if "ubi_participation_reduction_factor" in ui_params.get("labor_market", {}): # Ensure nested dict is updated
    final_params["labor_market"]["ubi_participation_reduction_factor"].update(ui_params["labor_market"]["ubi_participation_reduction_factor"])
final_params["economy"].update(ui_params.get("economy", {}))
final_params["government_finance"].update(ui_params.get("government_finance", {}))
# Note: data_paths, demographics, social_indicators are not currently UI configurable, so they'll use defaults.


# --- Execução da Simulação com Parâmetros Atuais ---
@st.cache_data
def run_sim(params_tuple_items): # Cache requer argumentos hashable
    params_dict = {}
    for k, v_or_tuple in params_tuple_items:
        if isinstance(v_or_tuple, tuple) and len(v_or_tuple) > 0 and isinstance(v_or_tuple[0], tuple): # Nested dict
            params_dict[k] = dict(v_or_tuple)
        else:
            params_dict[k] = v_or_tuple
            
    simulator = UBISimulatorRealisticV3(params=params_dict) # Use the new simulator
    try:
        simulator.run_simulation()
        return simulator
    except Exception as e:
        st.error(f"Erro durante a execução da simulação: {e}")
        # Return a simulator with empty results to prevent breaking the UI
        empty_simulator = UBISimulatorRealisticV3(params=params_dict)
        empty_simulator.results = {
            "Baseline (Sem RBU)": pd.DataFrame(columns=list(formatters_dict.keys())), # Use new formatters_dict
            "Com RBU": pd.DataFrame(columns=list(formatters_dict.keys()))
        }
        return empty_simulator


# Converte dict final para tupla de itens ordenados para ser hashable pelo cache
try:
    # Tenta converter valores complexos (como numpy arrays se existirem) para tipos básicos
    items_to_tuple = []
    for k, v in final_params.items():
        if isinstance(v, (np.ndarray, pd.DataFrame, pd.Series)):
             items_to_tuple.append((k, str(v))) # Converte para string como fallback
        else:
             items_to_tuple.append((k, v))
    params_tuple = tuple(sorted(items_to_tuple))
except TypeError as e:
    st.error(f"Erro ao criar tupla de parâmetros para cache: {e}. Alguns parâmetros podem não ser 'hashable'.")
    # Tenta converter todos os valores para string como fallback extremo
    params_tuple = tuple(sorted({k: str(v) for k,v in final_params.items()}.items()))


simulator = run_sim(params_tuple)
results = simulator.get_results()

# --- Exibição dos Resultados ---

# Verifica se os resultados são DataFrames válidos
if not isinstance(results, dict) or \
   "Baseline (Sem RBU)" not in results or not isinstance(results["Baseline (Sem RBU)"], pd.DataFrame) or \
   "Com RBU" not in results or not isinstance(results["Com RBU"], pd.DataFrame):
    st.error("Resultados da simulação inválidos ou não foram gerados.")
    # Attempt to show some info even on error
    st.write("Parâmetros Finais Usados:", final_params)
    st.stop()


baseline_df = results["Baseline (Sem RBU)"]
ubi_df = results["Com RBU"]

# Verifica se DataFrames não estão vazios antes de prosseguir
if baseline_df.empty or ubi_df.empty:
     st.warning("A simulação retornou resultados vazios. Verifique os parâmetros, especialmente os anos de início/fim e a configuração do país.")
     # Cria DFs vazios com colunas para evitar erros na plotagem
     sim_start = final_params.get("simulation_years", {}).get("start", datetime.date.today().year)
     sim_end = final_params.get("simulation_years", {}).get("end", sim_start + 1)
     baseline_df = pd.DataFrame(columns=formatters_dict.keys(), index=pd.RangeIndex(start=sim_start, stop=sim_end+1))
     ubi_df = pd.DataFrame(columns=formatters_dict.keys(), index=pd.RangeIndex(start=sim_start, stop=sim_end+1))
     baseline_df = baseline_df.fillna(0.0)
     ubi_df = ubi_df.fillna(0.0)


st.header(f"📊 Resultados da Simulação para {country_selected.title()}")

# --- Tabela Comparativa ---
st.subheader("Tabela Comparativa Anual")
_sim_start_year_final = int(final_params["simulation_years"]["start"])
_sim_end_year_final = int(final_params["simulation_years"]["end"])
_ubi_start_year_final = int(final_params["ubi"]["start_year"])

try:
    default_year_table = min(_sim_end_year_final, _ubi_start_year_final + 10)
    default_year_table = max(_sim_start_year_final, default_year_table)
except Exception: 
    default_year_table = _sim_end_year_final

year_to_display = st.slider(
    "Selecione o ano para a tabela comparativa:",
    min_value=_sim_start_year_final,
    max_value=_sim_end_year_final,
    value=int(default_year_table),
    step=1,
    key=f"compare_year_slider_{country_selected}"
)

summary_df = simulator.get_summary_dataframe(year_to_display)
if not summary_df.empty:
    # Use a more robust way to format, only applying to existing columns
    valid_formatters = {k: v for k, v in formatters_dict.items() if k in summary_df.columns or k in summary_df.index}
    try:
        st.dataframe(summary_df.style.format(formatter=valid_formatters, na_rep='-'), use_container_width=True)
    except Exception as e:
        st.warning(f"Erro ao formatar tabela de resumo: {e}. Exibindo dados brutos.")
        st.dataframe(summary_df, use_container_width=True)

else:
    st.warning(f"Não há dados disponíveis para a tabela no ano {year_to_display}.")


# --- Gráficos Comparativos ---
st.subheader("📈 Gráficos Comparativos ao Longo do Tempo")

indicators_to_plot = list(formatters_dict.keys()) # Use new formatters
# Filter out cohort-specific population/participation rates for general overview graphs
indicators_to_plot = [ind for ind in indicators_to_plot if not (ind.startswith("Pop ") or ind.startswith("Part Rate "))]


sim_ubi_start_final = int(final_params["ubi"]["start_year"])
cols = st.columns(2)
col_idx = 0

# Garante que os índices dos dataframes existem e são compatíveis antes de plotar
if not baseline_df.index.equals(ubi_df.index):
    st.warning("Índices dos cenários Baseline e Com RBU não são idênticos. Gráficos podem estar desalinhados.")
    # Tenta reindexar para alinhar, preenchendo com NaN onde não há dados
    common_index = baseline_df.index.union(ubi_df.index)
    baseline_df = baseline_df.reindex(common_index).fillna(0.0) # Ou ffill/bfill?
    ubi_df = ubi_df.reindex(common_index).fillna(0.0)


for indicator in indicators_to_plot:
    if indicator not in baseline_df.columns or indicator not in ubi_df.columns:
        # st.warning(f"Indicador '{indicator}' não encontrado para plotagem.")
        continue # Pula indicador se não estiver presente em ambos

    chart_data_baseline = baseline_df[indicator]
    chart_data_ubi = ubi_df[indicator]

    chart_df_data = {
        'Sem RBU (Baseline)': chart_data_baseline,
        'Com RBU': chart_data_ubi
        }
    chart_data = pd.DataFrame(chart_df_data)

    # Verifica se há dados não-NaN para plotar
    if not chart_data.isnull().all().all() and not chart_data.empty:
        current_col = cols[col_idx % 2]
        with current_col:
            st.markdown(f"**{indicator}**")
            st.line_chart(chart_data)
            if sim_ubi_start_final <= _sim_end_year_final:
                 st.caption(f"Início da RBU (ano {sim_ubi_start_final}) não visualmente marcado.")
        col_idx += 1
    # else:
        # st.warning(f"Sem dados válidos para plotar o indicador '{indicator}'.")


# --- Dados Completos (para Download) ---
st.subheader("💾 Dados Completos da Simulação")
with st.expander("Ver/Baixar Tabelas de Dados"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cenário Baseline (Sem RBU)**")
        st.dataframe(baseline_df.style.format(formatter=formatters_dict, na_rep='-'))
        st.download_button(
           "Download Baseline CSV", baseline_df.to_csv(index=True).encode('utf-8'),
           "baseline_results.csv", "text/csv", key='download-baseline'
         )
    with col2:
        st.markdown("**Cenário Com RBU**")
        st.dataframe(ubi_df.style.format(formatter=formatters_dict, na_rep='-'))
        st.download_button(
           "Download RBU CSV", ubi_df.to_csv(index=True).encode('utf-8'),
           "ubi_results.csv", "text/csv", key='download-ubi'
         )