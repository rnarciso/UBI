import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Default Parameters (Now including Inflation) ---
DEFAULT_PARAMS = {
    # Parâmetros da RBU
    "ubi_annual_amount_eur": 14400,  # Ex: 1200 €/mês * 12
    "ubi_start_year": 2027, # Adjusted default for demo
    "ubi_eligible_population_share": 1.0, # 1.0 = Universal

    # Parâmetros Econômicos e Comportamentais
    "labor_participation_reduction_factor": 0.04, # Redução líquida de 4% na taxa de participação
    "labor_hours_reduction_factor": 0.01, # Redução adicional de 1% nas horas médias
    "productivity_growth_boost_pp": 0.001, # Aumento de 0.1 p.p. no crescimento anual da produtividade
    "consumption_propensity_ubi_recipients": 0.8, # 80% da RBU gasta
    "consumption_reduction_high_income_financing": 0.2, # Redução consumo dos mais ricos se financiamento por impostos
    "innovation_startup_boost_factor": 1.05, # Aumento de 5% na taxa de criação de startups (simplificado)

    # Parâmetros Sociais
    "wellbeing_satisfaction_boost": 0.5, # Aumento na satisfação média de vida (escala 0-10)
    "mental_health_improvement_factor": 0.6, # Redução de 40% na prevalência de problemas (fator 0.6)
    "crime_reduction_per_gini_point": 0.05, # Redução de 5% no crime para cada 0.01 ponto de queda no Gini
    "unpaid_work_increase_factor": 1.1, # Aumento de 10% nas horas de trabalho não remunerado

    # Parâmetros de Financiamento e Reação
    "financing_model": "mixed_taxes_debt", # Opções: "progressive_tax", "wealth_tax", "debt", "mixed_taxes_debt"
    "additional_tax_rate_for_ubi": 0.05, # Alíquota adicional média (se financiado por imposto)
    "capital_flight_factor_high_tax": 0.005, # 0.5% da riqueza pode sair anualmente se impostos altos
    "migration_net_increase_factor_restricted": 1.1, # Aumento de 10% na imigração líquida (com regras)
    "migration_net_increase_factor_open": 1.5, # Aumento de 50% (hipotético, sem regras - não usado por default)

    # Parâmetros de Inflação (NOVOS)
    "baseline_annual_inflation_rate": 0.02, # Inflação anual base (ex: 2%)
    "inflation_ubi_demand_sensitivity": 0.15, # Aumento da inflação por ponto % de aumento Consumo/PIB via RBU
    "inflation_productivity_dampening_factor": 0.5, # Redução da inflação por ponto % de crescimento da produtividade

    # Parâmetros de Simulação
    "start_year": 2024,
    "end_year": 2075,

    # Dados Iniciais (Exemplo Simplificado - Usar dados reais!)
    "initial_population_millions": 84.0,
    "initial_gdp_trillion_eur": 4.0,
    "initial_participation_rate": 0.75,
    "initial_avg_hours_worked_per_worker": 1600,
    "initial_unemployment_rate": 0.055,
    "initial_gini_coefficient": 0.29,
    "initial_poverty_rate": 0.15,
    "initial_wellbeing_satisfaction": 7.0,
    "initial_mental_health_prevalence": 0.10,
    "initial_crime_rate_index": 100,
    "initial_net_migration_thousands": 100,
    "initial_millionaires_thousands": 150,
    "initial_govt_debt_gdp_ratio": 0.60,
    "baseline_annual_pop_growth_rate": 0.001,
    "baseline_annual_gdp_growth_rate": 0.012,
    "baseline_annual_productivity_growth_rate": 0.01,
    "poverty_floor": 0.03 # Piso mínimo para taxa de pobreza
}


class UBISimulator:
    """
    Simulador de Impactos de Longo Prazo da Renda Básica Universal (RBU).
    Modela cenários 'Com RBU' vs. 'Sem RBU' (Baseline) ao longo do tempo.
    """
    def __init__(self, params=None):
        if params is None:
             params = DEFAULT_PARAMS.copy() # Use default if none provided
        self.params = params
        # Garante que end_year seja pelo menos start_year
        self.params["end_year"] = max(params["start_year"], params["end_year"])
        self.years = np.arange(params["start_year"], params["end_year"] + 1)
        self.results = {} # Armazenará os DataFrames de resultados

    def _initialize_scenario(self):
        """Cria um DataFrame para armazenar os resultados anuais de um cenário."""
        indicators = [
            # Demografia e RBU
            "Population (Millions)", "UBI Cost (Billion EUR)",
            # Mercado de Trabalho
            "Labor Participation Rate", "Avg Hours Worked", "Unemployment Rate", "Labor Force (Millions)",
            # Economia e Finanças
            "GDP Real (Trillion EUR)", "GDP Growth Rate", "Productivity Growth Rate", "Productivity Level (Index)",
            "Consumption (Trillion EUR)", "Investment (Trillion EUR)", "Govt Debt/GDP Ratio",
            "Additional Tax Revenue (Billion EUR)", "Capital Flight Index",
            # Social
            "Gini Coefficient", "Poverty Rate", "Wellbeing Satisfaction (0-10)",
            "Mental Health Prevalence", "Crime Rate Index", "Unpaid Work Index",
            # Migração e Riqueza
            "Net Migration (Thousands)", "Millionaires (Thousands)",
            # Inflação (NOVO)
            "Inflation Rate"
        ]
        # Garante que temos pelo menos o ano inicial no índice se start == end
        index_years = self.years if len(self.years) > 0 else [self.params["start_year"]]
        df = pd.DataFrame(index=index_years, columns=indicators, dtype=float)

        # Preenche valores iniciais para o ano base (start_year - 1) para cálculos
        base_year = self.params["start_year"] - 1
        df.loc[base_year, "Population (Millions)"] = self.params["initial_population_millions"]
        df.loc[base_year, "GDP Real (Trillion EUR)"] = self.params["initial_gdp_trillion_eur"]
        df.loc[base_year, "Labor Participation Rate"] = self.params["initial_participation_rate"]
        df.loc[base_year, "Avg Hours Worked"] = self.params["initial_avg_hours_worked_per_worker"]
        df.loc[base_year, "Unemployment Rate"] = self.params["initial_unemployment_rate"]
        df.loc[base_year, "Productivity Level (Index)"] = 100.0 # Base 100
        df.loc[base_year, "Gini Coefficient"] = self.params["initial_gini_coefficient"]
        df.loc[base_year, "Poverty Rate"] = self.params["initial_poverty_rate"]
        df.loc[base_year, "Wellbeing Satisfaction (0-10)"] = self.params["initial_wellbeing_satisfaction"]
        df.loc[base_year, "Mental Health Prevalence"] = self.params["initial_mental_health_prevalence"]
        df.loc[base_year, "Crime Rate Index"] = self.params["initial_crime_rate_index"]
        df.loc[base_year, "Net Migration (Thousands)"] = self.params["initial_net_migration_thousands"]
        df.loc[base_year, "Millionaires (Thousands)"] = self.params["initial_millionaires_thousands"]
        df.loc[base_year, "Govt Debt/GDP Ratio"] = self.params["initial_govt_debt_gdp_ratio"]
        df.loc[base_year, "Capital Flight Index"] = 100.0
        df.loc[base_year, "Unpaid Work Index"] = 100.0
        df.loc[base_year, "Inflation Rate"] = self.params["baseline_annual_inflation_rate"] # Inflação inicial

        # Valores derivados iniciais
        pop_base = df.loc[base_year, "Population (Millions)"]
        part_rate_base = df.loc[base_year, "Labor Participation Rate"]
        unemp_rate_base = df.loc[base_year, "Unemployment Rate"]
        if pop_base > 0 and pd.notna(part_rate_base) and pd.notna(unemp_rate_base):
            initial_labor_force = pop_base * part_rate_base
            df.loc[base_year, "Labor Force (Millions)"] = initial_labor_force * (1 - unemp_rate_base) # Empregados
        else:
            df.loc[base_year, "Labor Force (Millions)"] = 0

        gdp_base = df.loc[base_year, "GDP Real (Trillion EUR)"]
        if gdp_base > 0:
            df.loc[base_year, "Consumption (Trillion EUR)"] = gdp_base * 0.6 # Suposição inicial
            df.loc[base_year, "Investment (Trillion EUR)"] = gdp_base * 0.2 # Suposição inicial
        else:
            df.loc[base_year, "Consumption (Trillion EUR)"] = 0
            df.loc[base_year, "Investment (Trillion EUR)"] = 0

        return df

    def run_simulation(self):
        """Executa a simulação para os cenários Baseline e Com RBU."""
        if len(self.years) == 0: # Caso start_year == end_year
            # Apenas inicializa e retorna os dados do ano base como se fosse o único ano
            baseline_df = self._initialize_scenario().loc[[self.params["start_year"]-1]].rename(index={self.params["start_year"]-1: self.params["start_year"]})
            ubi_df = self._initialize_scenario().loc[[self.params["start_year"]-1]].rename(index={self.params["start_year"]-1: self.params["start_year"]})
            baseline_df = baseline_df.fillna(0) # Preenche NaNs com 0 para evitar erros
            ubi_df = ubi_df.fillna(0)
            self.results["Baseline (Sem RBU)"] = baseline_df
            self.results["Com RBU"] = ubi_df
            return # Não há loop para executar

        # --- Cenário Baseline (Sem RBU) ---
        baseline_df = self._initialize_scenario()
        for year in self.years:
            prev_year = year - 1

            # --- Cálculos do Ano Atual (Baseline) ---
            prev_pop = baseline_df.loc[prev_year, "Population (Millions)"] if pd.notna(baseline_df.loc[prev_year, "Population (Millions)"]) else 0
            prev_mig = baseline_df.loc[prev_year, "Net Migration (Thousands)"] if pd.notna(baseline_df.loc[prev_year, "Net Migration (Thousands)"]) else 0
            baseline_df.loc[year, "Population (Millions)"] = prev_pop * (1 + self.params["baseline_annual_pop_growth_rate"]) + (prev_mig / 1000)
            baseline_df.loc[year, "Net Migration (Thousands)"] = prev_mig # Assume constante no baseline

            prev_part_rate = baseline_df.loc[prev_year, "Labor Participation Rate"] if pd.notna(baseline_df.loc[prev_year, "Labor Participation Rate"]) else 0
            prev_hours = baseline_df.loc[prev_year, "Avg Hours Worked"] if pd.notna(baseline_df.loc[prev_year, "Avg Hours Worked"]) else 0
            prev_unemp = baseline_df.loc[prev_year, "Unemployment Rate"] if pd.notna(baseline_df.loc[prev_year, "Unemployment Rate"]) else 0
            baseline_df.loc[year, "Labor Participation Rate"] = prev_part_rate # Constante no baseline
            baseline_df.loc[year, "Avg Hours Worked"] = prev_hours # Constante no baseline
            baseline_df.loc[year, "Unemployment Rate"] = prev_unemp # Constante no baseline (simplificado)

            prev_prod_level = baseline_df.loc[prev_year, "Productivity Level (Index)"] if pd.notna(baseline_df.loc[prev_year, "Productivity Level (Index)"]) else 100
            baseline_df.loc[year, "Productivity Growth Rate"] = self.params["baseline_annual_productivity_growth_rate"]
            baseline_df.loc[year, "Productivity Level (Index)"] = prev_prod_level * (1 + baseline_df.loc[year, "Productivity Growth Rate"])

            current_pop = baseline_df.loc[year, "Population (Millions)"] if pd.notna(baseline_df.loc[year, "Population (Millions)"]) else 0
            current_part_rate = baseline_df.loc[year, "Labor Participation Rate"]
            current_unemp_rate = baseline_df.loc[year, "Unemployment Rate"]
            if pd.notna(current_pop) and pd.notna(current_part_rate) and pd.notna(current_unemp_rate):
                potential_labor_force = current_pop * current_part_rate
                employed_labor_force = potential_labor_force * (1 - current_unemp_rate)
                baseline_df.loc[year, "Labor Force (Millions)"] = employed_labor_force
            else:
                 baseline_df.loc[year, "Labor Force (Millions)"] = 0

            gdp_prev = baseline_df.loc[prev_year, "GDP Real (Trillion EUR)"] if pd.notna(baseline_df.loc[prev_year, "GDP Real (Trillion EUR)"]) else 0
            # GDP growth based on labor and productivity changes (more robust)
            prev_labor = baseline_df.loc[prev_year, "Labor Force (Millions)"] if pd.notna(baseline_df.loc[prev_year, "Labor Force (Millions)"]) else 0
            current_labor = baseline_df.loc[year, "Labor Force (Millions)"]
            prev_hours_b = baseline_df.loc[prev_year, "Avg Hours Worked"] if pd.notna(baseline_df.loc[prev_year, "Avg Hours Worked"]) else 0
            current_hours_b = baseline_df.loc[year, "Avg Hours Worked"]
            current_prod_growth_b = baseline_df.loc[year, "Productivity Growth Rate"]

            if gdp_prev > 0 and prev_labor > 0 and prev_hours_b > 0 and pd.notna(current_labor) and pd.notna(current_hours_b) and pd.notna(current_prod_growth_b):
                labor_input_growth = (current_labor * current_hours_b) / (prev_labor * prev_hours_b) - 1
                gdp_current = gdp_prev * (1 + labor_input_growth + current_prod_growth_b)
            elif gdp_prev > 0: # Fallback to baseline growth if labor data missing
                gdp_current = gdp_prev * (1 + self.params["baseline_annual_gdp_growth_rate"])
            else:
                gdp_current = 0

            baseline_df.loc[year, "GDP Real (Trillion EUR)"] = gdp_current if gdp_current > 0 else 0
            baseline_df.loc[year, "GDP Growth Rate"] = (gdp_current / gdp_prev - 1) if gdp_prev else 0

            # --- Baseline Social Indicators (Assume simple trends or constant for baseline) ---
            prev_gini = baseline_df.loc[prev_year, "Gini Coefficient"] if pd.notna(baseline_df.loc[prev_year, "Gini Coefficient"]) else 0
            prev_poverty = baseline_df.loc[prev_year, "Poverty Rate"] if pd.notna(baseline_df.loc[prev_year, "Poverty Rate"]) else 0
            prev_wellbeing = baseline_df.loc[prev_year, "Wellbeing Satisfaction (0-10)"] if pd.notna(baseline_df.loc[prev_year, "Wellbeing Satisfaction (0-10)"]) else 0
            prev_mental = baseline_df.loc[prev_year, "Mental Health Prevalence"] if pd.notna(baseline_df.loc[prev_year, "Mental Health Prevalence"]) else 0
            prev_crime = baseline_df.loc[prev_year, "Crime Rate Index"] if pd.notna(baseline_df.loc[prev_year, "Crime Rate Index"]) else 0
            prev_unpaid = baseline_df.loc[prev_year, "Unpaid Work Index"] if pd.notna(baseline_df.loc[prev_year, "Unpaid Work Index"]) else 0
            baseline_df.loc[year, "Gini Coefficient"] = prev_gini # Constant baseline
            baseline_df.loc[year, "Poverty Rate"] = prev_poverty # Constant baseline
            baseline_df.loc[year, "Wellbeing Satisfaction (0-10)"] = prev_wellbeing # Constant baseline
            baseline_df.loc[year, "Mental Health Prevalence"] = prev_mental # Constant baseline
            baseline_df.loc[year, "Crime Rate Index"] = prev_crime # Constant baseline
            baseline_df.loc[year, "Unpaid Work Index"] = prev_unpaid # Constant baseline

            prev_millionaires = baseline_df.loc[prev_year, "Millionaires (Thousands)"] if pd.notna(baseline_df.loc[prev_year, "Millionaires (Thousands)"]) else 0
            prev_debt_ratio = baseline_df.loc[prev_year, "Govt Debt/GDP Ratio"] if pd.notna(baseline_df.loc[prev_year, "Govt Debt/GDP Ratio"]) else 0
            baseline_df.loc[year, "Millionaires (Thousands)"] = prev_millionaires * 1.02 # Simple growth baseline
            # Baseline debt ratio changes only by GDP growth denominator effect
            if gdp_current > 0 and gdp_prev > 0:
                 baseline_df.loc[year, "Govt Debt/GDP Ratio"] = prev_debt_ratio * (gdp_prev / gdp_current)
            else:
                 baseline_df.loc[year, "Govt Debt/GDP Ratio"] = prev_debt_ratio

            baseline_df.loc[year, "Capital Flight Index"] = 100.0 # No flight in baseline
            baseline_df.loc[year, "UBI Cost (Billion EUR)"] = 0
            baseline_df.loc[year, "Additional Tax Revenue (Billion EUR)"] = 0

            # Baseline Consumption/Investment follows GDP (simplistic)
            current_gdp_b = baseline_df.loc[year, "GDP Real (Trillion EUR)"]
            prev_cons_b = baseline_df.loc[prev_year, "Consumption (Trillion EUR)"] if pd.notna(baseline_df.loc[prev_year, "Consumption (Trillion EUR)"]) else 0
            prev_inv_b = baseline_df.loc[prev_year, "Investment (Trillion EUR)"] if pd.notna(baseline_df.loc[prev_year, "Investment (Trillion EUR)"]) else 0
            gdp_growth_b = baseline_df.loc[year, "GDP Growth Rate"]
            baseline_df.loc[year, "Consumption (Trillion EUR)"] = prev_cons_b * (1 + gdp_growth_b) if current_gdp_b > 0 else 0
            baseline_df.loc[year, "Investment (Trillion EUR)"] = prev_inv_b * (1 + gdp_growth_b) if current_gdp_b > 0 else 0

            # --- Baseline Inflation ---
            # Incorporates productivity dampening
            prod_growth_b = baseline_df.loc[year, "Productivity Growth Rate"] if pd.notna(baseline_df.loc[year, "Productivity Growth Rate"]) else 0
            inflation_reduction_from_prod_b = prod_growth_b * self.params["inflation_productivity_dampening_factor"]
            baseline_inflation = self.params["baseline_annual_inflation_rate"] - inflation_reduction_from_prod_b
            baseline_df.loc[year, "Inflation Rate"] = max(0, baseline_inflation) # Inflation floor at 0%


        self.results["Baseline (Sem RBU)"] = baseline_df.drop(self.params["start_year"] - 1).fillna(0) # Remove ano base e preenche NaNs

        # --- Cenário Com RBU ---
        ubi_df = self._initialize_scenario()
        for year in self.years:
            prev_year = year - 1
            ubi_active = year >= self.params["ubi_start_year"]

            # --- Cálculos do Ano Atual (Com RBU) ---
            migration_factor = 1.0
            if ubi_active:
                # Simplificado: usar apenas o fator restrito por enquanto
                migration_factor = self.params["migration_net_increase_factor_restricted"]
            prev_mig_ubi = ubi_df.loc[prev_year, "Net Migration (Thousands)"] if pd.notna(ubi_df.loc[prev_year, "Net Migration (Thousands)"]) else 0
            net_migration_ubi = self.params["initial_net_migration_thousands"] * migration_factor # Apply factor to initial value for simplicity
            ubi_df.loc[year, "Net Migration (Thousands)"] = net_migration_ubi

            prev_pop_ubi = ubi_df.loc[prev_year, "Population (Millions)"] if pd.notna(ubi_df.loc[prev_year, "Population (Millions)"]) else 0
            # Use baseline pop growth + calculated net migration
            ubi_df.loc[year, "Population (Millions)"] = prev_pop_ubi * (1 + self.params["baseline_annual_pop_growth_rate"]) + (net_migration_ubi / 1000)

            ubi_cost = 0
            current_pop_ubi = ubi_df.loc[year, "Population (Millions)"] if pd.notna(ubi_df.loc[year, "Population (Millions)"]) else 0
            if ubi_active and current_pop_ubi > 0 and self.params["ubi_annual_amount_eur"] > 0:
                eligible_pop = current_pop_ubi * self.params["ubi_eligible_population_share"]
                ubi_cost = (eligible_pop * 1_000_000 * self.params["ubi_annual_amount_eur"]) / 1_000_000_000
            ubi_df.loc[year, "UBI Cost (Billion EUR)"] = ubi_cost

            # --- Mercado de Trabalho (Com RBU) ---
            participation_rate_base_ubi = ubi_df.loc[prev_year, "Labor Participation Rate"] if pd.notna(ubi_df.loc[prev_year, "Labor Participation Rate"]) else 0
            avg_hours_base_ubi = ubi_df.loc[prev_year, "Avg Hours Worked"] if pd.notna(ubi_df.loc[prev_year, "Avg Hours Worked"]) else 0
            participation_rate_ubi = participation_rate_base_ubi * (1 - self.params["labor_participation_reduction_factor"]) if ubi_active else participation_rate_base_ubi
            avg_hours_ubi = avg_hours_base_ubi * (1 - self.params["labor_hours_reduction_factor"]) if ubi_active else avg_hours_base_ubi
            ubi_df.loc[year, "Labor Participation Rate"] = participation_rate_ubi
            ubi_df.loc[year, "Avg Hours Worked"] = avg_hours_ubi
            # Assume desemprego segue baseline por simplicidade (poderia ser afetado)
            prev_unemp_ubi = ubi_df.loc[prev_year, "Unemployment Rate"] if pd.notna(ubi_df.loc[prev_year, "Unemployment Rate"]) else 0
            ubi_df.loc[year, "Unemployment Rate"] = prev_unemp_ubi

            # --- Produtividade (Com RBU) ---
            productivity_growth_base_ubi = self.params["baseline_annual_productivity_growth_rate"]
            productivity_boost = self.params["productivity_growth_boost_pp"] if ubi_active else 0
            ubi_df.loc[year, "Productivity Growth Rate"] = productivity_growth_base_ubi + productivity_boost
            prev_prod_level_ubi = ubi_df.loc[prev_year, "Productivity Level (Index)"] if pd.notna(ubi_df.loc[prev_year, "Productivity Level (Index)"]) else 100
            ubi_df.loc[year, "Productivity Level (Index)"] = prev_prod_level_ubi * (1 + ubi_df.loc[year, "Productivity Growth Rate"])

            # --- Força de Trabalho (Com RBU) ---
            current_unemp_rate_ubi = ubi_df.loc[year, "Unemployment Rate"]
            if pd.notna(current_pop_ubi) and pd.notna(participation_rate_ubi) and pd.notna(current_unemp_rate_ubi):
                 potential_labor_force_ubi = current_pop_ubi * participation_rate_ubi
                 employed_labor_force_ubi = potential_labor_force_ubi * (1 - current_unemp_rate_ubi)
                 ubi_df.loc[year, "Labor Force (Millions)"] = employed_labor_force_ubi
            else:
                 ubi_df.loc[year, "Labor Force (Millions)"] = 0


            # --- GDP (Com RBU) ---
            gdp_prev_ubi = ubi_df.loc[prev_year, "GDP Real (Trillion EUR)"] if pd.notna(ubi_df.loc[prev_year, "GDP Real (Trillion EUR)"]) else 0
            prev_labor_ubi = ubi_df.loc[prev_year, "Labor Force (Millions)"] if pd.notna(ubi_df.loc[prev_year, "Labor Force (Millions)"]) else 0
            prev_hours_ubi = ubi_df.loc[prev_year, "Avg Hours Worked"] if pd.notna(ubi_df.loc[prev_year, "Avg Hours Worked"]) else 0
            current_prod_growth_ubi = ubi_df.loc[year, "Productivity Growth Rate"]
            current_labor_ubi = ubi_df.loc[year, "Labor Force (Millions)"]
            current_hours_ubi = ubi_df.loc[year, "Avg Hours Worked"]

            if gdp_prev_ubi > 0 and prev_labor_ubi > 0 and prev_hours_ubi > 0 and pd.notna(current_labor_ubi) and pd.notna(current_hours_ubi) and pd.notna(current_prod_growth_ubi):
                labor_input_growth_ubi = (current_labor_ubi * current_hours_ubi) / (prev_labor_ubi * prev_hours_ubi) - 1 if (prev_labor_ubi * prev_hours_ubi) > 0 else 0
                gdp_current_ubi = gdp_prev_ubi * (1 + labor_input_growth_ubi + current_prod_growth_ubi)
            elif gdp_prev_ubi > 0: # Fallback
                 gdp_current_ubi = gdp_prev_ubi * (1 + self.params["baseline_annual_gdp_growth_rate"])
            else:
                 gdp_current_ubi = 0

            # --- Financiamento e Dívida (Com RBU) ---
            tax_revenue = 0
            debt_increase_ratio = 0
            current_gdp_ubi = gdp_current_ubi # Use calculated GDP for this year

            if ubi_active and current_gdp_ubi > 0 and ubi_cost > 0:
                # Calculate tax based on *current* GDP if tax-based financing
                if self.params["financing_model"] in ["progressive_tax", "wealth_tax", "mixed_taxes_debt"]:
                    # Simplified: Apply flat additional rate to current GDP
                    tax_revenue = current_gdp_ubi * 1000 * self.params["additional_tax_rate_for_ubi"]

                ubi_df.loc[year, "Additional Tax Revenue (Billion EUR)"] = tax_revenue

                # Calculate unfunded portion and debt impact
                unfunded_cost = ubi_cost - tax_revenue # In Billions EUR
                if unfunded_cost > 0 and self.params["financing_model"] in ["debt", "mixed_taxes_debt"]:
                    debt_increase_ratio = (unfunded_cost / 1000) / current_gdp_ubi # As ratio of GDP
                elif unfunded_cost < 0: # If taxes exceed cost, assume it reduces debt
                     debt_increase_ratio = (unfunded_cost / 1000) / current_gdp_ubi

            else: # No UBI active or zero cost/GDP
                 ubi_df.loc[year, "Additional Tax Revenue (Billion EUR)"] = 0


            prev_debt_ratio_ubi = ubi_df.loc[prev_year, "Govt Debt/GDP Ratio"] if pd.notna(ubi_df.loc[prev_year, "Govt Debt/GDP Ratio"]) else 0
            # Update debt ratio: Previous debt value + new debt issuance, divided by current GDP
            if current_gdp_ubi > 0 and gdp_prev_ubi > 0:
                 # (Previous Debt / Current GDP) + (New Debt / Current GDP)
                 current_debt_ratio = (prev_debt_ratio_ubi * gdp_prev_ubi / current_gdp_ubi) + debt_increase_ratio
            elif current_gdp_ubi > 0: # Handle first year case or if prev GDP was zero
                 current_debt_ratio = prev_debt_ratio_ubi + debt_increase_ratio
            else: # If current GDP is zero, debt ratio becomes infinite/undefined, keep previous
                 current_debt_ratio = prev_debt_ratio_ubi

            ubi_df.loc[year, "Govt Debt/GDP Ratio"] = max(0, current_debt_ratio) # Prevent negative debt ratio

            # --- Consumo e Investimento (Com RBU) ---
            prev_consumption_ubi = ubi_df.loc[prev_year, "Consumption (Trillion EUR)"] if pd.notna(ubi_df.loc[prev_year, "Consumption (Trillion EUR)"]) else 0
            prev_investment_ubi = ubi_df.loc[prev_year, "Investment (Trillion EUR)"] if pd.notna(ubi_df.loc[prev_year, "Investment (Trillion EUR)"]) else 0
            gdp_growth_ubi = (current_gdp_ubi / gdp_prev_ubi - 1) if gdp_prev_ubi else 0

            # Start with base growth
            consumption_ubi = prev_consumption_ubi * (1 + gdp_growth_ubi)
            investment_ubi = prev_investment_ubi * (1 + gdp_growth_ubi)
            net_consumption_boost_trillions = 0 # Initialize

            if ubi_active and ubi_cost > 0:
                # Calculate net effect of UBI spending vs. tax impact on consumption
                # Use tax_revenue calculated earlier (in Billions)
                consumption_boost_from_ubi = ubi_cost * self.params["consumption_propensity_ubi_recipients"]
                consumption_reduction_from_taxes = (tax_revenue * self.params["consumption_reduction_high_income_financing"]) if self.params["financing_model"] != "debt" else 0
                net_consumption_boost_billions = consumption_boost_from_ubi - consumption_reduction_from_taxes
                net_consumption_boost_trillions = net_consumption_boost_billions / 1000

                consumption_ubi += net_consumption_boost_trillions

                # Simple investment reduction factor if financed by taxes
                if self.params["financing_model"] != "debt":
                     investment_ubi *= (1 - self.params["additional_tax_rate_for_ubi"] * 0.1) # Small reduction effect

            ubi_df.loc[year, "Consumption (Trillion EUR)"] = consumption_ubi if consumption_ubi > 0 else 0
            ubi_df.loc[year, "Investment (Trillion EUR)"] = investment_ubi if investment_ubi > 0 else 0

            # Update GDP based on C+I+G+(X-M) changes - simplified, just ensure C+I doesn't exceed calculated GDP for now
            # This is a simplification; a full macro model would recalculate GDP based on C, I, G changes.
            # For now, we stick with the production-side GDP calculation.
            ubi_df.loc[year, "GDP Real (Trillion EUR)"] = gdp_current_ubi if gdp_current_ubi > 0 else 0
            ubi_df.loc[year, "GDP Growth Rate"] = gdp_growth_ubi

            # --- Indicadores Sociais (Com RBU) ---
            gini_base_ubi = ubi_df.loc[prev_year, "Gini Coefficient"] if pd.notna(ubi_df.loc[prev_year, "Gini Coefficient"]) else 0
            poverty_base_ubi = ubi_df.loc[prev_year, "Poverty Rate"] if pd.notna(ubi_df.loc[prev_year, "Poverty Rate"]) else 0
            wellbeing_base_ubi = ubi_df.loc[prev_year, "Wellbeing Satisfaction (0-10)"] if pd.notna(ubi_df.loc[prev_year, "Wellbeing Satisfaction (0-10)"]) else 0
            mental_health_base_ubi = ubi_df.loc[prev_year, "Mental Health Prevalence"] if pd.notna(ubi_df.loc[prev_year, "Mental Health Prevalence"]) else 0
            crime_base_ubi = ubi_df.loc[prev_year, "Crime Rate Index"] if pd.notna(ubi_df.loc[prev_year, "Crime Rate Index"]) else 0
            unpaid_work_base_ubi = ubi_df.loc[prev_year, "Unpaid Work Index"] if pd.notna(ubi_df.loc[prev_year, "Unpaid Work Index"]) else 0

            if ubi_active:
                # Gini: Assume linear reduction based on UBI amount (relative to a reference like 14400)
                gini_reduction_max = 0.04 # Max reduction for reference amount
                ubi_amount_effect = self.params["ubi_annual_amount_eur"] / 14400 if self.params["ubi_annual_amount_eur"] > 0 else 0
                gini_reduction = gini_reduction_max * ubi_amount_effect if self.params["ubi_annual_amount_eur"] > 0 else 0
                gini_ubi = max(0, gini_base_ubi - gini_reduction)
                ubi_df.loc[year, "Gini Coefficient"] = gini_ubi

                # Poverty: Assume significant reduction, towards a floor
                poverty_reduction_factor = 0.9 # Strong reduction effect
                poverty_ubi = poverty_base_ubi * (1 - poverty_reduction_factor * ubi_amount_effect) # Scaled by UBI amount
                ubi_df.loc[year, "Poverty Rate"] = max(self.params.get("poverty_floor", 0.03), poverty_ubi)

                # Wellbeing: Direct boost
                ubi_df.loc[year, "Wellbeing Satisfaction (0-10)"] = min(10, wellbeing_base_ubi + self.params["wellbeing_satisfaction_boost"] * ubi_amount_effect) # Scaled

                # Mental Health: Improvement factor
                ubi_df.loc[year, "Mental Health Prevalence"] = mental_health_base_ubi * (1 - (1 - self.params["mental_health_improvement_factor"]) * ubi_amount_effect) # Scaled improvement

                # Crime: Reduction linked to Gini change
                gini_change = gini_base_ubi - gini_ubi
                crime_reduction_factor = (gini_change / 0.01) * self.params["crime_reduction_per_gini_point"] if gini_change > 0 else 0
                ubi_df.loc[year, "Crime Rate Index"] = crime_base_ubi * (1 - crime_reduction_factor) if crime_base_ubi > 0 else 0

                # Unpaid Work: Increase factor
                ubi_df.loc[year, "Unpaid Work Index"] = unpaid_work_base_ubi * self.params["unpaid_work_increase_factor"]
            else: # If UBI not active, values remain same as previous year
                ubi_df.loc[year, "Gini Coefficient"] = gini_base_ubi
                ubi_df.loc[year, "Poverty Rate"] = poverty_base_ubi
                ubi_df.loc[year, "Wellbeing Satisfaction (0-10)"] = wellbeing_base_ubi
                ubi_df.loc[year, "Mental Health Prevalence"] = mental_health_base_ubi
                ubi_df.loc[year, "Crime Rate Index"] = crime_base_ubi
                ubi_df.loc[year, "Unpaid Work Index"] = unpaid_work_base_ubi

            # --- Riqueza e Fuga de Capital (Com RBU) ---
            millionaires_base_ubi = ubi_df.loc[prev_year, "Millionaires (Thousands)"] if pd.notna(ubi_df.loc[prev_year, "Millionaires (Thousands)"]) else 0
            capital_flight_base_ubi = ubi_df.loc[prev_year, "Capital Flight Index"] if pd.notna(ubi_df.loc[prev_year, "Capital Flight Index"]) else 100
            millionaires_ubi = millionaires_base_ubi * 1.02 # Base growth
            capital_flight_ubi = capital_flight_base_ubi

            # Apply capital flight if UBI active and financed by non-debt means
            if ubi_active and self.params["financing_model"] != "debt" and self.params["capital_flight_factor_high_tax"] > 0:
                flight_increase_factor = 1 + self.params["capital_flight_factor_high_tax"]
                capital_flight_ubi = capital_flight_base_ubi * flight_increase_factor # Index increases
                millionaires_ubi *= (1 - self.params["capital_flight_factor_high_tax"]) # Stock decreases

            ubi_df.loc[year, "Millionaires (Thousands)"] = max(0, millionaires_ubi)
            ubi_df.loc[year, "Capital Flight Index"] = capital_flight_ubi if capital_flight_ubi > 0 else 0

            # --- Inflação (Com RBU) ---
            current_inflation = self.params["baseline_annual_inflation_rate"] # Start with baseline

            # 1. Productivity Dampening Effect
            current_prod_growth_ubi = ubi_df.loc[year, "Productivity Growth Rate"] if pd.notna(ubi_df.loc[year, "Productivity Growth Rate"]) else 0
            inflation_reduction_from_prod = current_prod_growth_ubi * self.params["inflation_productivity_dampening_factor"]
            current_inflation -= inflation_reduction_from_prod

            # 2. UBI Demand Effect (if active and positive GDP)
            if ubi_active and current_gdp_ubi > 0 and net_consumption_boost_trillions > 0:
                # Calculate boost in consumption relative to GDP
                ubi_consumption_boost_ratio = net_consumption_boost_trillions / current_gdp_ubi
                inflation_boost_from_demand = ubi_consumption_boost_ratio * self.params["inflation_ubi_demand_sensitivity"]
                current_inflation += inflation_boost_from_demand

            # Ensure inflation doesn't go below a certain floor (e.g., 0%)
            ubi_df.loc[year, "Inflation Rate"] = max(0, current_inflation)


        self.results["Com RBU"] = ubi_df.drop(self.params["start_year"] - 1).fillna(0) # Remove ano base e preenche NaNs

    def get_results(self):
        """Retorna os DataFrames de resultados para ambos os cenários."""
        return self.results

    def get_summary_dataframe(self, year):
        """Retorna um DataFrame formatado para a tabela comparativa de um ano específico."""
        if not self.results or "Baseline (Sem RBU)" not in self.results or "Com RBU" not in self.results :
             # Retorna DataFrame vazio se não houver resultados
             return pd.DataFrame(columns=["Indicador", "Cenário Sem RBU", "Cenário Com RBU"])

        target_year = max(self.params["start_year"], min(year, self.params["end_year"])) # Garante que o ano está no intervalo

        if target_year not in self.results["Baseline (Sem RBU)"].index or target_year not in self.results["Com RBU"].index:
             st.error(f"Ano {target_year} fora do intervalo da simulação ({self.params['start_year']}-{self.params['end_year']}) ou dados ausentes.")
             # Retorna DataFrame vazio
             return pd.DataFrame(columns=["Indicador", "Cenário Sem RBU", "Cenário Com RBU"])


        baseline_data = self.results["Baseline (Sem RBU)"].loc[target_year]
        ubi_data = self.results["Com RBU"].loc[target_year]

        summary = pd.DataFrame({
            "Indicador": baseline_data.index,
            "Cenário Sem RBU": baseline_data.values,
            "Cenário Com RBU": ubi_data.values
        })

        # Formatação para melhor leitura
        formatters = {
            "Population (Millions)": "{:.1f}",
            "UBI Cost (Billion EUR)": "{:.1f}",
            "Labor Participation Rate": "{:.1%}",
            "Avg Hours Worked": "{:.0f}",
            "Unemployment Rate": "{:.1%}",
            "Labor Force (Millions)": "{:.1f}",
            "GDP Real (Trillion EUR)": "{:.2f}",
            "GDP Growth Rate": "{:.2%}",
            "Productivity Growth Rate": "{:.2%}",
            "Productivity Level (Index)": "{:.1f}",
            "Consumption (Trillion EUR)": "{:.2f}",
            "Investment (Trillion EUR)": "{:.2f}",
            "Govt Debt/GDP Ratio": "{:.1%}",
            "Additional Tax Revenue (Billion EUR)": "{:.1f}",
            "Capital Flight Index": "{:.1f}",
            "Gini Coefficient": "{:.3f}",
            "Poverty Rate": "{:.1%}",
            "Wellbeing Satisfaction (0-10)": "{:.1f}",
            "Mental Health Prevalence": "{:.2%}",
            "Crime Rate Index": "{:.1f}",
            "Unpaid Work Index": "{:.1f}",
            "Net Migration (Thousands)": "{:.0f}",
            "Millionaires (Thousands)": "{:.0f}",
            "Inflation Rate": "{:.2%}" # Formatter for Inflation
        }

        formatted_summary = summary.copy()
        # Aplica formatação apenas se a coluna existir e não for NaN
        for indicator_name, fmt in formatters.items():
             mask = formatted_summary["Indicador"] == indicator_name
             if mask.any():
                 idx = formatted_summary[mask].index
                 for col in ["Cenário Sem RBU", "Cenário Com RBU"]:
                     # Tenta formatar, trata erros (ex: valor não numérico)
                     try:
                         # Ensure value is numeric before formatting
                         values_to_format = pd.to_numeric(formatted_summary.loc[idx, col], errors='coerce')
                         formatted_summary.loc[idx, col] = values_to_format.apply(
                             lambda x: fmt.format(x) if pd.notna(x) else ('0' if pd.notna(formatted_summary.loc[idx, col].iloc[0]) else 'N/A') # Handle potential NaNs after coercion
                         )
                     except (ValueError, TypeError, AttributeError) as e:
                          # st.warning(f"Could not format {indicator_name} in {col}. Value: {formatted_summary.loc[idx, col].iloc[0]}. Error: {e}")
                          formatted_summary.loc[idx, col] = 'Error' # Indicate formatting error


        return formatted_summary.set_index("Indicador")


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

# Cria dicionário para armazenar parâmetros da UI
current_params = {}

# --- Grupo: Configuração da RBU ---
with st.sidebar.expander("🔵 Configuração da RBU", expanded=True):
    current_params["ubi_annual_amount_eur"] = st.slider(
        "Valor Anual da RBU (€/pessoa)", 0, 30000, DEFAULT_PARAMS["ubi_annual_amount_eur"], 500, format="€%d", key="ubi_amount"
    )
    # Garante que o start year não ultrapasse o end year menos 1 (para ter pelo menos 1 ano de simulação)
    max_start_year = DEFAULT_PARAMS["end_year"] - 1 if DEFAULT_PARAMS["end_year"] > DEFAULT_PARAMS["start_year"] else DEFAULT_PARAMS["start_year"]
    current_params["ubi_start_year"] = st.slider(
        "Ano de Início da RBU", DEFAULT_PARAMS["start_year"], max_start_year, DEFAULT_PARAMS["ubi_start_year"], 1, key="ubi_start"
    )
    current_params["ubi_eligible_population_share"] = st.slider(
        "Parcela da População Elegível (%)", 0.0, 1.0, DEFAULT_PARAMS["ubi_eligible_population_share"], 0.05, format="%.0f%%", key="ubi_eligibility"
    )

# --- Grupo: Impactos Econômicos e Comportamentais ---
with st.sidebar.expander("💼 Econômicos e Comportamentais"):
    current_params["labor_participation_reduction_factor"] = st.slider(
        "Redução na Participação no Trabalho (%)", 0.0, 0.20, DEFAULT_PARAMS["labor_participation_reduction_factor"], 0.005, format="%.1f%%", key="labor_part_reduc"
    )
    current_params["labor_hours_reduction_factor"] = st.slider(
        "Redução nas Horas Médias Trabalhadas (%)", 0.0, 0.10, DEFAULT_PARAMS["labor_hours_reduction_factor"], 0.005, format="%.1f%%", key="labor_hours_reduc"
    )
    current_params["productivity_growth_boost_pp"] = st.slider(
        "Impulso Adicional na Produtividade (p.p./ano)", 0.0, 0.01, DEFAULT_PARAMS["productivity_growth_boost_pp"], 0.0005, format="%.3f p.p.", key="prod_boost"
    )
    current_params["consumption_propensity_ubi_recipients"] = st.slider(
        "Propensão a Consumir RBU (Beneficiários, %)", 0.0, 1.0, DEFAULT_PARAMS["consumption_propensity_ubi_recipients"], 0.05, format="%.0f%%", key="consump_prop_ubi"
    )
    current_params["consumption_reduction_high_income_financing"] = st.slider(
        "Redução Consumo Ricos (se taxados, % do imposto)", 0.0, 0.5, DEFAULT_PARAMS["consumption_reduction_high_income_financing"], 0.05, format="%.0f%%", key="consump_reduc_tax"
    )
    # Simplified innovation proxy adjustment
    # current_params["innovation_startup_boost_factor"] = st.slider(
    #     "Multiplicador de Novas Empresas (Índice, 1=base)", 0.8, 1.5, DEFAULT_PARAMS["innovation_startup_boost_factor"], 0.01, format="%.2f", key="innovation_boost"
    # ) # Commented out as it's not directly used in current calculations

# --- Grupo: Impactos Sociais ---
with st.sidebar.expander("🌍 Impactos Sociais"):
    current_params["wellbeing_satisfaction_boost"] = st.slider(
        "Aumento na Satisfação Média (0-10)", 0.0, 2.0, DEFAULT_PARAMS["wellbeing_satisfaction_boost"], 0.1, key="wellbeing_boost"
    )
    current_params["mental_health_improvement_factor"] = st.slider(
        "Fator de Melhoria Saúde Mental (1=sem mudança, 0=elimina)", 0.0, 1.0, DEFAULT_PARAMS["mental_health_improvement_factor"], 0.05, format="%.2f", key="mental_health_factor"
    )
    current_params["crime_reduction_per_gini_point"] = st.slider(
        "Redução Crime por Ponto Gini (0.01) (%)", 0.0, 0.10, DEFAULT_PARAMS["crime_reduction_per_gini_point"], 0.005, format="%.1f%%", key="crime_gini_reduc"
    )
    current_params["unpaid_work_increase_factor"] = st.slider(
        "Aumento Trabalho Não Remunerado (Índice, 1=base)", 1.0, 1.5, DEFAULT_PARAMS["unpaid_work_increase_factor"], 0.01, format="%.2f", key="unpaid_work_factor"
    )

# --- Grupo: Financiamento e Reações ---
with st.sidebar.expander("💰 Financiamento e Reações"):
    financing_options = ["mixed_taxes_debt", "progressive_tax", "wealth_tax", "debt"]
    current_params["financing_model"] = st.selectbox(
        "Modelo de Financiamento Principal",
        options=financing_options,
        index=financing_options.index(DEFAULT_PARAMS["financing_model"]), key="financing_model"
    )
    current_params["additional_tax_rate_for_ubi"] = st.slider(
        "Alíquota Adicional Média (se financiado por imposto, % do PIB)", 0.0, 0.15, DEFAULT_PARAMS["additional_tax_rate_for_ubi"], 0.005, format="%.1f%%", key="tax_rate_ubi"
    )
    current_params["capital_flight_factor_high_tax"] = st.slider(
        "Fuga de Capital Anual (se impostos altos, % da riqueza)", 0.0, 0.05, DEFAULT_PARAMS["capital_flight_factor_high_tax"], 0.001, format="%.2f%%", key="capital_flight"
    )
    current_params["migration_net_increase_factor_restricted"] = st.slider(
        "Aumento Imigração Líquida (c/ regras, Índice 1=base)", 1.0, 2.0, DEFAULT_PARAMS["migration_net_increase_factor_restricted"], 0.05, format="%.2f", key="migration_factor"
    )
    # current_params["migration_net_increase_factor_open"] = st.slider(
    #     "Aumento Imigração Líquida (sem regras, Índice 1=base)", 1.0, 3.0, DEFAULT_PARAMS["migration_net_increase_factor_open"], 0.1, format="%.1f", key="migration_factor_open"
    # ) # Less relevant for default restricted model

# --- Grupo: Inflação ---
with st.sidebar.expander("📈 Parâmetros de Inflação"):
    current_params["baseline_annual_inflation_rate"] = st.slider(
        "Inflação Anual Base (%)", 0.0, 0.10, DEFAULT_PARAMS["baseline_annual_inflation_rate"], 0.005, format="%.1f%%", key="base_inflation"
    )
    current_params["inflation_ubi_demand_sensitivity"] = st.slider(
        "Sensibilidade Inflação à Demanda RBU", 0.0, 0.5, DEFAULT_PARAMS["inflation_ubi_demand_sensitivity"], 0.01, format="%.2f", key="inflation_demand_sens"
    )
    current_params["inflation_productivity_dampening_factor"] = st.slider(
        "Redução Inflação por Produtividade", 0.0, 1.0, DEFAULT_PARAMS["inflation_productivity_dampening_factor"], 0.05, format="%.2f", key="inflation_prod_damp"
    )


# --- Grupo: Configuração da Simulação ---
with st.sidebar.expander("⏱️ Configuração da Simulação"):
     # Keep start/end year fixed for now, or make them careful inputs
    # current_params["start_year"] = st.number_input("Ano Inicial", min_value=2020, max_value=2050, value=DEFAULT_PARAMS["start_year"], step=1, key="sim_start_year")
    # current_params["end_year"] = st.number_input("Ano Final", min_value=current_params.get("start_year", DEFAULT_PARAMS["start_year"]) + 5, max_value=2100, value=DEFAULT_PARAMS["end_year"], step=1, key="sim_end_year")
    # Keep start/end years from DEFAULT_PARAMS for simplicity now
    current_params["start_year"] = DEFAULT_PARAMS["start_year"]
    current_params["end_year"] = DEFAULT_PARAMS["end_year"]
    st.write(f"Simulação de {current_params['start_year']} a {current_params['end_year']}")


# --- Grupo: Dados Iniciais (Avançado) ---
with st.sidebar.expander("📊 Dados Iniciais (Avançado)"):
    st.caption("Ajustar com cautela. Impactam todo o cenário base.")
    current_params["initial_population_millions"] = st.number_input("População Inicial (Milhões)", min_value=1.0, value=DEFAULT_PARAMS["initial_population_millions"], step=0.1, key="init_pop")
    current_params["initial_gdp_trillion_eur"] = st.number_input("PIB Inicial (Trilhões €)", min_value=0.1, value=DEFAULT_PARAMS["initial_gdp_trillion_eur"], step=0.1, key="init_gdp")
    current_params["initial_participation_rate"] = st.slider("Taxa de Participação Inicial (%)", 0.4, 0.9, DEFAULT_PARAMS["initial_participation_rate"], 0.01, format="%.0f%%", key="init_part_rate")
    current_params["initial_avg_hours_worked_per_worker"] = st.number_input("Horas Médias Anuais Iniciais / Trabalhador", min_value=1000, max_value=2500, value=DEFAULT_PARAMS["initial_avg_hours_worked_per_worker"], step=10, key="init_avg_hours")
    current_params["initial_unemployment_rate"] = st.slider("Taxa de Desemprego Inicial (%)", 0.01, 0.20, DEFAULT_PARAMS["initial_unemployment_rate"], 0.005, format="%.1f%%", key="init_unemp_rate")
    current_params["initial_gini_coefficient"] = st.slider("Coeficiente Gini Inicial", 0.2, 0.6, DEFAULT_PARAMS["initial_gini_coefficient"], 0.01, format="%.2f", key="init_gini")
    current_params["initial_poverty_rate"] = st.slider("Taxa de Pobreza Inicial (%)", 0.01, 0.50, DEFAULT_PARAMS["initial_poverty_rate"], 0.01, format="%.0f%%", key="init_poverty")
    current_params["initial_wellbeing_satisfaction"] = st.slider("Satisfação Inicial (0-10)", 0.0, 10.0, DEFAULT_PARAMS["initial_wellbeing_satisfaction"], 0.1, key="init_wellbeing")
    current_params["initial_mental_health_prevalence"] = st.slider("Prevalência Saúde Mental Inicial (%)", 0.01, 0.30, DEFAULT_PARAMS["initial_mental_health_prevalence"], 0.01, format="%.0f%%", key="init_mental")
    current_params["initial_crime_rate_index"] = st.number_input("Índice Criminalidade Inicial", min_value=10, value=DEFAULT_PARAMS["initial_crime_rate_index"], step=5, key="init_crime")
    current_params["initial_net_migration_thousands"] = st.number_input("Migração Líquida Inicial (Milhares)", min_value=-500, max_value=1000, value=DEFAULT_PARAMS["initial_net_migration_thousands"], step=10, key="init_migration")
    current_params["initial_millionaires_thousands"] = st.number_input("Milionários Iniciais (Milhares)", min_value=0, value=DEFAULT_PARAMS["initial_millionaires_thousands"], step=10, key="init_millionaires")
    current_params["initial_govt_debt_gdp_ratio"] = st.slider("Dívida/PIB Inicial (%)", 0.0, 2.0, DEFAULT_PARAMS["initial_govt_debt_gdp_ratio"], 0.05, format="%.0f%%", key="init_debt_ratio")
    current_params["baseline_annual_pop_growth_rate"] = st.slider("Cresc. Pop. Anual Base (%)", -0.01, 0.02, DEFAULT_PARAMS["baseline_annual_pop_growth_rate"], 0.001, format="%.2f%%", key="base_pop_growth")
    current_params["baseline_annual_gdp_growth_rate"] = st.slider("Cresc. PIB Real Anual Base (%)", -0.01, 0.05, DEFAULT_PARAMS["baseline_annual_gdp_growth_rate"], 0.001, format="%.2f%%", key="base_gdp_growth")
    current_params["baseline_annual_productivity_growth_rate"] = st.slider("Cresc. Produtividade Anual Base (%)", 0.0, 0.03, DEFAULT_PARAMS["baseline_annual_productivity_growth_rate"], 0.001, format="%.2f%%", key="base_prod_growth")
    current_params["poverty_floor"] = st.slider("Piso Taxa de Pobreza (%)", 0.0, 0.10, DEFAULT_PARAMS["poverty_floor"], 0.005, format="%.1f%%", key="poverty_floor")


# Add missing default params that weren't in sidebar (e.g., innovation factor if commented out)
for key, value in DEFAULT_PARAMS.items():
    if key not in current_params:
        current_params[key] = value

# --- Execução da Simulação com Parâmetros Atuais ---

# Usar cache pode ser útil para simulações mais longas
@st.cache_data # Cache the simulation run based on parameters
def run_sim(params):
    simulator = UBISimulator(params=params)
    simulator.run_simulation()
    return simulator

# Make params hashable for caching (convert dict to tuple of items)
params_tuple = tuple(sorted(current_params.items()))
simulator = run_sim(current_params) # Pass the dict, cache handles the tuple internally
results = simulator.get_results()

# --- Exibição dos Resultados ---

if not results or "Baseline (Sem RBU)" not in results or "Com RBU" not in results:
    st.error("Erro ao gerar resultados da simulação. Verifique os parâmetros e tente novamente.")
    st.stop() # Stop execution if results are bad

baseline_df = results["Baseline (Sem RBU)"]
ubi_df = results["Com RBU"]

st.header("📊 Resultados da Simulação")

# --- Tabela Comparativa ---
st.subheader("Tabela Comparativa Anual")
# Set default year more intelligently (e.g., 10 years after UBI start or end year)
default_year = min(current_params["end_year"], current_params["ubi_start_year"] + 10)

year_to_display = st.slider(
    "Selecione o ano para a tabela comparativa:",
    min_value=current_params["start_year"],
    max_value=current_params["end_year"],
    value=default_year,
    step=1,
    key="compare_year_slider"
)
summary_df = simulator.get_summary_dataframe(year_to_display)
if not summary_df.empty:
    st.dataframe(summary_df, use_container_width=True)
else:
    st.warning(f"Não foi possível gerar a tabela para o ano {year_to_display}.")


# --- Gráficos Comparativos ---
st.subheader("📈 Gráficos Comparativos ao Longo do Tempo")

indicators_to_plot = [
    "GDP Real (Trillion EUR)",
    "Inflation Rate", # Added
    "Labor Participation Rate",
    "Gini Coefficient",
    "Poverty Rate",
    "Wellbeing Satisfaction (0-10)",
    "Govt Debt/GDP Ratio",
    "UBI Cost (Billion EUR)",
    "Crime Rate Index"
]

# Determine ubi_start year from potentially adjusted params
sim_ubi_start = current_params["ubi_start_year"]

# Use columns for better layout
cols = st.columns(2)
col_idx = 0

for indicator in indicators_to_plot:
    current_col = cols[col_idx % 2]
    with current_col:
        if indicator in baseline_df.columns and indicator in ubi_df.columns:
            st.markdown(f"**{indicator}**")
            # Combine the series for the st.line_chart
            chart_data = pd.DataFrame({
                'Sem RBU (Baseline)': baseline_df[indicator],
                'Com RBU': ubi_df[indicator]
            })

            # Check if the indicator requires percentage formatting
            is_percentage = any(fmt in indicator.lower() for fmt in ['rate', 'ratio', 'gini', 'prevalence']) or '%' in formatters.get(indicator,"")

            st.line_chart(chart_data)
            # Add a note about the UBI start line (st.line_chart doesn't support vertical lines easily)
            if sim_ubi_start <= current_params["end_year"]:
                 st.caption(f"Início da RBU (ano {sim_ubi_start}) não visualmente marcado.")

        else:
            st.warning(f"Indicador '{indicator}' não encontrado nos resultados para plotagem.")
    col_idx += 1


# --- Dados Completos (para Download) ---
st.subheader("💾 Dados Completos da Simulação")
with st.expander("Ver/Baixar Tabelas de Dados"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cenário Baseline (Sem RBU)**")
        st.dataframe(baseline_df.style.format(formatter=formatters, na_rep='-')) # Apply basic formatting
        st.download_button(
           "Download Baseline CSV",
           baseline_df.to_csv().encode('utf-8'),
           "baseline_results.csv",
           "text/csv",
           key='download-baseline'
         )
    with col2:
        st.markdown("**Cenário Com RBU**")
        st.dataframe(ubi_df.style.format(formatter=formatters, na_rep='-')) # Apply basic formatting
        st.download_button(
           "Download RBU CSV",
           ubi_df.to_csv().encode('utf-8'),
           "ubi_results.csv",
           "text/csv",
           key='download-ubi'
         )