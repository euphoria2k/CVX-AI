import { ComponentBase } from '@syncfusion/ej2-vue-base';
export declare const properties: string[];
export declare const modelProps: string[];
/**
 * Represents Vuejs Smithchart Component
 * ```vue
 * <ejs-smithchart></ejs-smithchart>
 * ```
 */
export declare class SmithchartComponent extends ComponentBase {
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
    export(type: Object, fileName: string, orientation?: Object): void;
    mouseEnd(e: Object): void;
    mouseMove(e: Object): void;
    print(id?: string[] | string | Object): void;
    smithchartOnClick(e: Object): void;
    smithchartOnResize(e: Object): boolean;
}
export declare const SmithchartPlugin: {
    name: string;
    install(Vue: any): void;
};
