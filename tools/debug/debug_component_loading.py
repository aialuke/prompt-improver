"""
Debug component loading to find which component is failing to load.
"""
import asyncio
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def debug_component_loading():
    """Test loading each tier individually."""
    print('🔍 Debugging Component Loading by Tier...')
    from prompt_improver.ml.orchestration.integration.direct_component_loader import ComponentTier, DirectComponentLoader
    loader = DirectComponentLoader()
    for tier in ComponentTier:
        print(f'\n📦 Testing {tier.value}:')
        tier_components = await loader.load_tier_components(tier)
        expected = len(loader.component_paths[tier])
        loaded = len(tier_components)
        print(f'  Expected: {expected}, Loaded: {loaded}')
        if loaded < expected:
            print(f'  ❌ Missing {expected - loaded} components:')
            for comp_name in loader.component_paths[tier]:
                if comp_name not in tier_components:
                    print(f'    - {comp_name}')
        else:
            print(f'  ✅ All components loaded successfully')
if __name__ == '__main__':
    asyncio.run(debug_component_loading())
