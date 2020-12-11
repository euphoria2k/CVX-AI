import { ComponentBase } from '@syncfusion/ej2-vue-base';
export declare const properties: string[];
export declare const modelProps: string[];
/**
 * Represents the Essential JS 2 VueJS FileManager Component.
 * ```vue
 * <ejs-filemanager showThumbnail='false'></ejs-filemanager>
 * ```
 */
export declare class FileManagerComponent extends ComponentBase {
    ej2Instances: any;
    propKeys: string[];
    models: string[];
    hasChildDirective: boolean;
    protected hasInjectedModules: boolean;
    tagMapper: {
        [key: string]: Object;
    };
    tagNameMapper: Object;
    constructor();
    setProperties(prop: any, muteOnChange: boolean): void;
    render(createElement: any): any;
    clearSelection(): void;
    createFolder(name?: string): void;
    deleteFiles(ids?: string[]): void;
    disableMenuItems(items: string[]): void;
    disableToolbarItems(items: string[]): void;
    downloadFiles(ids?: string[]): void;
    enableToolbarItems(items: string[]): void;
    filterFiles(filterData?: Object): void;
    getMenuItemIndex(item: string): number;
    getSelectedFiles(): Object[];
    getToolbarItemIndex(item: string): number;
    openFile(id: string): void;
    refreshFiles(): void;
    refreshLayout(): void;
    renameFile(id?: string, name?: string): void;
    selectAll(): void;
    uploadFiles(): void;
}
export declare const FileManagerPlugin: {
    name: string;
    install(Vue: any): void;
};
