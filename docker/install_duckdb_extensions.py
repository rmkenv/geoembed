#!/usr/bin/env python3
"""
Pre-install DuckDB extensions to avoid runtime network issues.
This is especially important for ARM64 containers that may have network connectivity issues.
"""

import duckdb

def install_extension(conn, extension_name):
    """Install and load a DuckDB extension with error handling."""
    try:
        conn.execute(f'INSTALL {extension_name}')
        conn.execute(f'LOAD {extension_name}')
        print(f'Successfully pre-installed {extension_name} extension')
        return True
    except Exception as e:
        print(f'Warning: Could not pre-install {extension_name} extension: {e}')
        return False

def main():
    """Main function to install DuckDB extensions."""
    conn = duckdb.connect(':memory:')
    
    extensions = ['spatial', 'json']
    success_count = 0
    
    for extension in extensions:
        if install_extension(conn, extension):
            success_count += 1
    
    conn.close()
    print(f'Pre-installation complete: {success_count}/{len(extensions)} extensions installed successfully')

if __name__ == '__main__':
    main()