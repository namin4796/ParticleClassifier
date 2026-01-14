import os
import sys

# 1. Where is this script running?
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"📍 Script is here:    {current_dir}")

# 2. Where are we trying to look?
# We go one level up (..), then into HybridEngine, then into build
target_path = os.path.abspath(os.path.join(current_dir, '../HybridEngine/build'))
print(f"🔍 Looking for lib in: {target_path}")

# 3. Does that folder exist?
if os.path.exists(target_path):
    print("✅ Folder EXISTS!")
    print(f"📂 Files inside '{target_path}':")
    found_lib = False
    for f in os.listdir(target_path):
        print(f"   - {f}")
        if "gravity_core" in f and (".so" in f or ".pyd" in f):
            found_lib = True
    
    if found_lib:
        print("\n🎉 SUCCESS: Found the compiled library file!")
    else:
        print("\n⚠️  WARNING: Folder exists, but I don't see 'gravity_core.so' or '.pyd' here.")
        print("    Did you forget to run 'cmake --build .' inside HybridEngine/build?")
else:
    print("❌ Folder NOT found.")
    
    # 4. Debug: What DOES exist one level up?
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    print(f"\nHere is what actually exists in '{parent_dir}':")
    try:
        for f in os.listdir(parent_dir):
            print(f"   - {f}")
    except:
        print("   (Could not list parent directory)")
