def find_package_versions():
    with open('requirements.in', 'r') as in_file:
        packages_in = in_file.readlines()

    with open('requirements.txt', 'r') as txt_file:
        packages_txt = txt_file.readlines()

    result = {}
    for package_in in packages_in:
        package_in = package_in.strip()
        for package_txt in packages_txt:
            if package_txt.startswith(package_in):
                result[package_in] = package_txt.strip().split('==')[1]
                break

    for key, value in result.items():
        print(f'{key}=={value}')

if __name__ == '__main__':
    find_package_versions()