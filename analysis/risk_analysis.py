import pandas as pd
import numpy as np
from scipy import stats

class RiskAnalyzer:
    def risk_assessment(self, opportunities):
        """Perform comprehensive risk assessment on investment opportunities"""
        
        if opportunities is None or opportunities.empty:
            print("No opportunities to analyze")
            return
        
        try:
            # Initialize risk metrics DataFrame
            risk_metrics = pd.DataFrame()
            
            # 1. Volatility Risk (24h price change)
            risk_metrics['volatility'] = opportunities['price_change_24h'].abs()
            
            # 2. Market Cap Risk (inverse log of market cap)
            risk_metrics['market_cap_risk'] = 1 / np.log1p(opportunities['market_cap'])
            
            # 3. Volume Risk (based on volume to market cap ratio)
            risk_metrics['volume_risk'] = 1 - (opportunities['total_volume'] / opportunities['market_cap'])
            
            # 4. Price Stability Risk (using 7d change if available)
            if 'price_change_7d' in opportunities.columns:
                risk_metrics['stability_risk'] = opportunities['price_change_7d'].abs() / 7
            
            # 5. Market Position Risk (based on market cap rank)
            risk_metrics['position_risk'] = opportunities['market_cap_rank'] / 1000
            
            # 6. Momentum Risk (comparing 24h to 7d trend)
            if 'price_change_7d' in opportunities.columns:
                risk_metrics['momentum_risk'] = abs(
                    opportunities['price_change_24h'] - 
                    (opportunities['price_change_7d'] / 7)
                )
            
            # Calculate weighted total risk
            weights = {
                'volatility': 0.25,
                'market_cap_risk': 0.20,
                'volume_risk': 0.15,
                'stability_risk': 0.15,
                'position_risk': 0.15,
                'momentum_risk': 0.10
            }
            
            # Calculate total risk score
            risk_metrics['total_risk'] = sum(
                risk_metrics[metric] * weight 
                for metric, weight in weights.items() 
                if metric in risk_metrics.columns
            )
            
            # Assign risk levels
            risk_metrics['risk_level'] = pd.qcut(
                risk_metrics['total_risk'], 
                q=3, 
                labels=['Low', 'Medium', 'High']
            )
            
            # Print detailed risk assessment
            print("\n🎯 Comprehensive Risk Assessment")
            print("=" * 50)
            
            for idx, row in opportunities.iterrows():
                print(f"\n{row['name']} ({row['symbol'].upper()})")
                print("-" * 30)
                print(f"Overall Risk Level: {risk_metrics.loc[idx, 'risk_level']}")
                print(f"Risk Score: {risk_metrics.loc[idx, 'total_risk']:.4f}")
                print("\nRisk Breakdown:")
                
                # Print individual risk components
                for metric in weights.keys():
                    if metric in risk_metrics.columns:
                        print(f"- {metric.replace('_', ' ').title()}: "
                              f"{risk_metrics.loc[idx, metric]:.4f}")
                
                print(f"\nKey Metrics:")
                print(f"- 24h Volatility: {abs(row['price_change_24h']):.2f}%")
                print(f"- Market Cap Rank: #{row['market_cap_rank']}")
                if 'price_change_7d' in opportunities.columns:
                    print(f"- 7d Price Change: {row['price_change_7d']:.2f}%")
                print("-" * 30)
            
            return risk_metrics
            
        except Exception as e:
            print(f"Error in risk assessment: {e}")
            return None
