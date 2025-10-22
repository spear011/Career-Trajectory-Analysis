"""
Occupation Transition Network Construction

Based on research proposal:
- Nodes: Occupations (O*NET-SOC 2019 codes)
  - Static attributes: O*NET categories
  - Time-varying attributes: Employment count, wage
- Edges: Directed transitions between occupations
  - Time-varying attributes: transition counts, upward mobility indicators
- Temporal granularity: quarterly or annual

Study Period: 2017-2024
- Pre-Pandemic: 2017-2019 (Baseline mobility patterns)
- COVID Shock: Mar 2020-2021 (Labor market volatility)
- Post-Pandemic Recovery: 2022-2024 (Persistence of changes)
"""

import pandas as pd
import networkx as nx
from datetime import datetime
import pickle
import json
from pathlib import Path
from collections import defaultdict

# Constants
OCCUPATION_COLUMN = 'ONET_2019'  # O*NET-SOC 2019 code
OCCUPATION_NAME_COLUMN = 'ONET_2019_NAME'

# Study period (as defined in research proposal)
STUDY_START_YEAR = 2017
STUDY_END_YEAR = 2024

def load_data():
    """Load data"""
    print("Loading data...")
    
    job_df = pd.read_csv('/common/home/users/c/chhan/Work/Career-Trajectory-Analysis/data/job_group_0.csv')
    wage_df = pd.read_csv('/common/home/users/c/chhan/Work/Career-Trajectory-Analysis/data/wage_interpolated_1999_2022_soc2019_unique.csv')
    
    print(f"  Job records: {len(job_df):,}")
    print(f"  Wage records: {len(wage_df):,}")
    print(f"  Study period: {STUDY_START_YEAR}-{STUDY_END_YEAR}")
    
    return job_df, wage_df

def parse_date(date_str):
    """Parse date string (YYYY-MM format)"""
    if pd.isna(date_str) or date_str == '':
        return None
    try:
        return datetime.strptime(date_str, '%Y-%m')
    except:
        return None

def get_year_from_date(date_str):
    """Extract year from date string"""
    date = parse_date(date_str)
    return date.year if date else None

def prepare_job_data(job_df):
    """Prepare job data"""
    print("\nPreparing job data...")
    
    # Parse dates
    job_df['start_year'] = job_df['JOB_START_DATE'].apply(get_year_from_date)
    job_df['end_year'] = job_df['JOB_END_DATE'].apply(get_year_from_date)
    
    # Remove Unclassified occupations
    job_df = job_df[job_df[OCCUPATION_COLUMN] != '99-9999.00'].copy()
    
    # Select required columns only
    cols = ['ID', OCCUPATION_COLUMN, OCCUPATION_NAME_COLUMN, 
            'start_year', 'end_year', 'IS_CURRENT',
            'SOC_EMSI_2019_2', 'SOC_EMSI_2019_2_NAME']
    job_df = job_df[cols].copy()
    
    print(f"  Valid job records: {len(job_df):,}")
    print(f"  Unique users: {job_df['ID'].nunique():,}")
    print(f"  Unique occupations: {job_df[OCCUPATION_COLUMN].nunique():,}")
    
    return job_df

def prepare_wage_data(wage_df):
    """Prepare wage data - convert to O*NET code format"""
    print("\nPreparing wage data...")
    
    # Convert OCC_CODE to O*NET format (SOC -> O*NET)
    # Example: '11-1011' -> '11-1011.00'
    wage_df['ONET_CODE'] = wage_df['OCC_CODE'].apply(
        lambda x: f"{x}.00" if pd.notna(x) and '.' not in str(x) else x
    )
    
    # Filter wage data to study period
    wage_df = wage_df[
        (wage_df['year'] >= STUDY_START_YEAR) & 
        (wage_df['year'] <= STUDY_END_YEAR)
    ].copy()
    
    # Calculate average wages by year and occupation (national average)
    wage_summary = wage_df.groupby(['year', 'ONET_CODE']).agg({
        'A_MEAN': 'mean',      # Annual mean wage
        'A_MEDIAN': 'mean',    # Annual median wage
        'TOT_EMP': 'sum'       # Total employment
    }).reset_index()
    
    wage_summary.columns = ['year', 'occupation', 'mean_wage', 'median_wage', 'employment']
    
    print(f"  Wage records: {len(wage_summary):,}")
    print(f"  Year range: {wage_summary['year'].min()}-{wage_summary['year'].max()}")
    print(f"  Occupations with wage data: {wage_summary['occupation'].nunique():,}")
    
    return wage_summary

def build_user_career_paths(job_df):
    """Build career paths for each user"""
    print("\nBuilding user career paths...")
    
    # Sort by user ID and start year
    job_df_sorted = job_df.sort_values(['ID', 'start_year']).copy()
    
    # Build paths for each user
    user_paths = defaultdict(list)
    
    for user_id, group in job_df_sorted.groupby('ID'):
        for _, row in group.iterrows():
            user_paths[user_id].append({
                'occupation': row[OCCUPATION_COLUMN],
                'occupation_name': row[OCCUPATION_NAME_COLUMN],
                'start_year': row['start_year'],
                'end_year': row['end_year'],
                'soc_2': row['SOC_EMSI_2019_2'],
                'soc_2_name': row['SOC_EMSI_2019_2_NAME']
            })
    
    print(f"  Career paths built for {len(user_paths):,} users")
    
    return user_paths

def extract_transitions(user_paths, min_year=STUDY_START_YEAR, max_year=STUDY_END_YEAR):
    """Extract transitions from career paths"""
    print("\nExtracting transitions...")
    print(f"  Study Period: {min_year}-{max_year}")
    
    transitions = []
    
    for user_id, path in user_paths.items():
        # Sort path by start year
        path_sorted = sorted([p for p in path if p['start_year'] is not None], 
                            key=lambda x: x['start_year'])
        
        # Extract transitions between consecutive jobs
        for i in range(len(path_sorted) - 1):
            from_job = path_sorted[i]
            to_job = path_sorted[i + 1]
            
            # Transition year (start year of to_job)
            transition_year = to_job['start_year']
            
            # Filter by year
            if min_year and transition_year < min_year:
                continue
            if max_year and transition_year > max_year:
                continue
            
            transitions.append({
                'user_id': user_id,
                'from_occupation': from_job['occupation'],
                'from_occupation_name': from_job['occupation_name'],
                'to_occupation': to_job['occupation'],
                'to_occupation_name': to_job['occupation_name'],
                'transition_year': transition_year,
                'from_year': from_job['start_year'],
                'to_year': to_job['start_year'],
                'from_soc_2': from_job['soc_2'],
                'to_soc_2': to_job['soc_2']
            })
    
    transitions_df = pd.DataFrame(transitions)
    
    if len(transitions_df) > 0:
        print(f"  Total transitions: {len(transitions_df):,}")
        print(f"  Unique users: {transitions_df['user_id'].nunique():,}")
        print(f"  Year range: {transitions_df['transition_year'].min()}-{transitions_df['transition_year'].max()}")
    else:
        print("  No transitions found")
    
    return transitions_df

def build_temporal_network(transitions_df, wage_df, period='annual'):
    """Build temporal network"""
    print("\nBuilding temporal network...")
    print(f"  Temporal granularity: {period}")
    
    networks = {}
    
    # Build network for each year
    for year in sorted(transitions_df['transition_year'].unique()):
        year_transitions = transitions_df[transitions_df['transition_year'] == year]
        year_wage = wage_df[wage_df['year'] == year]
        
        # 방향 그래프 생성
        G = nx.DiGraph()
        G.graph['year'] = year
        G.graph['period'] = period
        
        # 노드 추가: 해당 연도에 등장하는 모든 직업
        all_occupations = set(year_transitions['from_occupation'].unique()) | \
                         set(year_transitions['to_occupation'].unique())
        
        for occ in all_occupations:
            # 직업명 찾기
            occ_name = year_transitions[
                (year_transitions['from_occupation'] == occ) | 
                (year_transitions['to_occupation'] == occ)
            ].iloc[0]
            
            if year_transitions[year_transitions['from_occupation'] == occ].empty:
                name = occ_name['to_occupation_name']
            else:
                name = occ_name['from_occupation_name']
            
            # 노드 속성 설정
            node_attrs = {
                'occupation_code': occ,
                'occupation_name': name,
                'year': year
            }
            
            # 임금 정보 추가
            wage_info = year_wage[year_wage['occupation'] == occ]
            if not wage_info.empty:
                node_attrs['mean_wage'] = float(wage_info.iloc[0]['mean_wage'])
                node_attrs['median_wage'] = float(wage_info.iloc[0]['median_wage'])
                node_attrs['employment'] = float(wage_info.iloc[0]['employment'])
            else:
                node_attrs['mean_wage'] = None
                node_attrs['median_wage'] = None
                node_attrs['employment'] = None
            
            # 고용 수 계산 (해당 연도에 이 직업에서 전환한 사람 + 이 직업으로 전환한 사람)
            from_count = len(year_transitions[year_transitions['from_occupation'] == occ])
            to_count = len(year_transitions[year_transitions['to_occupation'] == occ])
            node_attrs['transition_volume'] = from_count + to_count
            
            G.add_node(occ, **node_attrs)
        
        # 엣지 추가: 직업 간 전환
        edge_counts = year_transitions.groupby(['from_occupation', 'to_occupation']).size()
        
        for (from_occ, to_occ), count in edge_counts.items():
            # 상향 이동 여부 판단 (임금 기준)
            from_wage = G.nodes[from_occ].get('mean_wage')
            to_wage = G.nodes[to_occ].get('mean_wage')
            
            upward_mobility = None
            if from_wage is not None and to_wage is not None:
                upward_mobility = to_wage > from_wage
                wage_change = to_wage - from_wage
                wage_change_pct = (wage_change / from_wage) * 100 if from_wage > 0 else 0
            else:
                wage_change = None
                wage_change_pct = None
            
            # 엣지 속성
            edge_attrs = {
                'transition_count': int(count),
                'year': year,
                'upward_mobility': upward_mobility,
                'wage_change': wage_change,
                'wage_change_pct': wage_change_pct
            }
            
            G.add_edge(from_occ, to_occ, **edge_attrs)
        
        networks[year] = G
        
        print(f"  Year {year}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return networks

def calculate_network_statistics(networks):
    """네트워크 통계 계산"""
    print("\nCalculating network statistics...")
    
    stats = []
    
    for year, G in sorted(networks.items()):
        # 기본 통계
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G) if n_nodes > 1 else 0
        
        # 상향 이동 비율
        upward_edges = sum(1 for _, _, d in G.edges(data=True) 
                          if d.get('upward_mobility') is True)
        downward_edges = sum(1 for _, _, d in G.edges(data=True) 
                            if d.get('upward_mobility') is False)
        upward_ratio = upward_edges / n_edges if n_edges > 0 else 0
        
        # 전환 볼륨
        total_transitions = sum(d['transition_count'] for _, _, d in G.edges(data=True))
        
        # 평균 차수
        avg_in_degree = sum(d for _, d in G.in_degree()) / n_nodes if n_nodes > 0 else 0
        avg_out_degree = sum(d for _, d in G.out_degree()) / n_nodes if n_nodes > 0 else 0
        
        stats.append({
            'year': year,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': density,
            'total_transitions': total_transitions,
            'upward_edges': upward_edges,
            'downward_edges': downward_edges,
            'upward_ratio': upward_ratio,
            'avg_in_degree': avg_in_degree,
            'avg_out_degree': avg_out_degree
        })
    
    stats_df = pd.DataFrame(stats)
    
    print("\nNetwork Statistics Summary:")
    print(stats_df.to_string(index=False))
    
    return stats_df

def save_networks(networks, stats_df, output_dir='./results'):
    """네트워크 저장"""
    print("\nSaving networks...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. 각 연도별 네트워크를 GraphML 형식으로 저장
    for year, G in networks.items():
        # NaN 연도 스킵
        if pd.isna(year):
            continue
            
        # None 값을 문자열로 변환 (GraphML 호환성)
        G_copy = G.copy()
        for node in G_copy.nodes():
            for key, value in G_copy.nodes[node].items():
                if value is None:
                    G_copy.nodes[node][key] = 'None'
        
        for u, v in G_copy.edges():
            for key, value in G_copy.edges[u, v].items():
                if value is None:
                    G_copy.edges[u, v][key] = 'None'
        
        filename = output_path / f'network_{int(year)}.graphml'
        nx.write_graphml(G_copy, filename)
        print(f"  Saved: {filename}")
    
    # 2. 전체 네트워크를 pickle로 저장 (Python 분석용)
    with open(output_path / 'networks_all.pkl', 'wb') as f:
        pickle.dump(networks, f)
    print(f"  Saved: {output_path / 'networks_all.pkl'}")
    
    # 3. 통계 저장
    stats_df.to_csv(output_path / 'network_statistics.csv', index=False)
    print(f"  Saved: {output_path / 'network_statistics.csv'}")
    
    # 4. 메타데이터 저장
    metadata = {
        'occupation_column': OCCUPATION_COLUMN,
        'occupation_name_column': OCCUPATION_NAME_COLUMN,
        'years': sorted(networks.keys()),
        'n_networks': len(networks),
        'total_nodes': sum(G.number_of_nodes() for G in networks.values()),
        'total_edges': sum(G.number_of_edges() for G in networks.values())
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {output_path / 'metadata.json'}")
    
    return output_path

def main():
    print("="*80)
    print("OCCUPATION TRANSITION NETWORK CONSTRUCTION")
    print("="*80)
    
    # 1. 데이터 로드
    job_df, wage_df = load_data()
    
    # 2. 데이터 전처리
    job_df = prepare_job_data(job_df)
    wage_df = prepare_wage_data(wage_df)
    
    # 3. 사용자 경로 구축
    user_paths = build_user_career_paths(job_df)
    
    # 4. 전환 추출
    transitions_df = extract_transitions(user_paths)
    
    # 5. 시계열 네트워크 구축
    networks = build_temporal_network(transitions_df, wage_df, period='annual')
    
    # 6. 네트워크 통계 계산
    stats_df = calculate_network_statistics(networks)
    
    # 7. 저장
    output_path = save_networks(networks, stats_df)
    
    print("\n" + "="*80)
    print("NETWORK CONSTRUCTION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_path}")
    print(f"Number of temporal networks: {len(networks)}")
    print(f"Year range: {min(networks.keys())}-{max(networks.keys())}")
    
    return networks, stats_df

if __name__ == '__main__':
    networks, stats = main()