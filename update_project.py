
import sys
import re

project_path = 'SortformerTest.xcodeproj/project.pbxproj'
backup_path = project_path + '.backup'

def update_pbxproj():
    try:
        with open(project_path, 'r') as f:
            content = f.read()
            
        with open(backup_path, 'w') as f:
            f.write(content)
            
        # Split by `buildSettings = {`
        parts = content.split('buildSettings = {')
        new_content = parts[0]
        
        for part in parts[1:]:
            # part starts with content of buildSettings, ends with `};` somewhere
            # Find the closing `};`
            end_idx = part.find('};')
            if end_idx == -1:
                new_content += 'buildSettings = {' + part
                continue
                
            block_content = part[:end_idx]
            rest = part[end_idx:]
            
            # Check if this is a configuration we want to modify (Debug/Release)
            # Actually, `buildSettings` blocks appear in XCBuildConfiguration objects.
            
            if 'HEADER_SEARCH_PATHS' in block_content:
                if 'SortformerTest/UMAP/include' not in block_content:
                    # Append to existing
                    if 'HEADER_SEARCH_PATHS = (' in block_content:
                        block_content = block_content.replace(
                            'HEADER_SEARCH_PATHS = (',
                            'HEADER_SEARCH_PATHS = (\n\t\t\t\t\t"$(SRCROOT)/SortformerTest/UMAP/include",'
                        )
                    else:
                        # Replace single entry with array
                        block_content = re.sub(
                            r'HEADER_SEARCH_PATHS = ([^;]+);',
                            r'HEADER_SEARCH_PATHS = (\n\t\t\t\t\t\1,\n\t\t\t\t\t"$(SRCROOT)/SortformerTest/UMAP/include",\n\t\t\t\t);',
                            block_content
                        )
            else:
                # Add new
                block_content = '\n\t\t\t\tHEADER_SEARCH_PATHS = "$(SRCROOT)/SortformerTest/UMAP/include";' + block_content
            
            new_content += 'buildSettings = {' + block_content + rest
            
        with open(project_path, 'w') as f:
            f.write(new_content)
            
        print("Successfully updated project.pbxproj")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    update_pbxproj()
